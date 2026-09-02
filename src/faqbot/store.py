"""Vector store, BM25 keyword index, and the hybrid retriever that fuses them.

Why hybrid, in one paragraph. Dense vectors are good at meaning and bad at
strings. Ask a support bot "is NW-FILT-02 compatible with the AR-1 Pro?" and a
pure embedding search will happily return the chunk about *filters in general*,
because ``NW-FILT-02`` is a rare token that contributes almost nothing to a
dense vector. BM25 does the opposite: it does not know that "runtime" and
"battery life" are related, but a rare exact token is the strongest signal it
has, so it nails the part number. Fusing the two ranked lists with Reciprocal
Rank Fusion gives a system that answers both kinds of question, and it does so
without needing a tuned score-scale mapping between two retrievers whose scores
are not comparable.

Everything here is pure standard library. ``numpy`` is used only, and
optionally, for ``.npz`` persistence of large indexes.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .embedding import Embedder, cosine
from .textutil import tokenize
from .types import Chunk, ScoredChunk

__all__ = [
    "VectorStore",
    "BM25Index",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "MetadataFilter",
]

MetadataFilter = Dict[str, Any]


def _matches(meta: Dict[str, Any], flt: Optional[MetadataFilter]) -> bool:
    """Metadata filter: equality, membership for list values, callables.

    ``{"kind": "faq_pair"}`` matches equality; ``{"kind": ["faq_pair", "html"]}``
    matches membership; ``{"score": lambda v: v > 3}`` matches a predicate.
    """
    if not flt:
        return True
    for key, want in flt.items():
        got = meta.get(key)
        if callable(want):
            if not want(got):
                return False
        elif isinstance(want, (list, tuple, set)):
            if got not in want:
                return False
        elif got != want:
            return False
    return True


class VectorStore:
    """In-memory dense index with JSON / npz persistence.

    Chunks are keyed by ``chunk_id`` and grouped by ``doc_id`` so that a
    document can be replaced or removed atomically. That grouping is what makes
    incremental updates safe: re-ingesting one edited page deletes exactly its
    old chunks instead of leaving stale ones behind to be retrieved forever.
    """

    FORMAT_VERSION = 1

    def __init__(self, dim: int, *, embedder_name: str = "hashing") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.embedder_name = embedder_name
        self._chunks: Dict[str, Chunk] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._by_doc: Dict[str, List[str]] = {}

    # -- inspection ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    @property
    def chunks(self) -> List[Chunk]:
        return [self._chunks[cid] for cid in self._chunks]

    @property
    def doc_ids(self) -> List[str]:
        return list(self._by_doc)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def vector(self, chunk_id: str) -> Optional[List[float]]:
        return self._vectors.get(chunk_id)

    def stats(self) -> Dict[str, Any]:
        sizes = [len(c.text) for c in self._chunks.values()]
        return {
            "chunks": len(self._chunks),
            "documents": len(self._by_doc),
            "dim": self.dim,
            "embedder": self.embedder_name,
            "mean_chunk_chars": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
            "max_chunk_chars": max(sizes) if sizes else 0,
        }

    # -- mutation --------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Add chunks with their vectors. Existing ``chunk_id``s are overwritten."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        added = 0
        for chunk, vec in zip(chunks, vectors):
            if len(vec) != self.dim:
                raise ValueError(
                    "vector for %s has dim %d, store dim is %d" % (chunk.chunk_id, len(vec), self.dim)
                )
            if chunk.chunk_id not in self._chunks:
                self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)
                added += 1
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = [float(v) for v in vec]
        return added

    def upsert_document(
        self, doc_id: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> Tuple[int, int]:
        """Replace every chunk of ``doc_id``. Returns ``(removed, added)``.

        This is the only correct way to re-index a changed document. Calling
        :meth:`add` alone leaves the previous chunking behind, and because the
        old chunks still embed well, they keep winning retrieval with content
        that no longer exists on the page.
        """
        removed = self.delete_document(doc_id)
        for chunk in chunks:
            if chunk.doc_id != doc_id:
                raise ValueError(
                    "chunk %s belongs to %s, not %s" % (chunk.chunk_id, chunk.doc_id, doc_id)
                )
        added = self.add(chunks, vectors)
        return removed, added

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks of a document. Returns how many were removed."""
        ids = self._by_doc.pop(doc_id, [])
        for cid in ids:
            self._chunks.pop(cid, None)
            self._vectors.pop(cid, None)
        return len(ids)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._by_doc.clear()

    # -- search ----------------------------------------------------------
    def search(
        self,
        query_vector: Sequence[float],
        k: int = 5,
        *,
        where: Optional[MetadataFilter] = None,
    ) -> List[ScoredChunk]:
        """Cosine top-k, optionally restricted by metadata.

        Ties are broken by ``chunk_id`` so results are stable across runs;
        unstable tie-breaking is a classic source of "the eval numbers moved
        and nothing changed".
        """
        if len(query_vector) != self.dim:
            raise ValueError("query dim %d != store dim %d" % (len(query_vector), self.dim))
        scored: List[Tuple[float, str]] = []
        for cid, chunk in self._chunks.items():
            if not _matches(chunk.meta, where):
                continue
            scored.append((cosine(query_vector, self._vectors[cid]), cid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        out: List[ScoredChunk] = []
        for rank, (score, cid) in enumerate(scored[: max(0, k)], start=1):
            out.append(
                ScoredChunk(
                    chunk=self._chunks[cid],
                    score=float(score),
                    rank=rank,
                    components={"dense": float(score)},
                )
            )
        return out

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.FORMAT_VERSION,
            "dim": self.dim,
            "embedder": self.embedder_name,
            "chunks": [self._chunks[cid].to_dict() for cid in self._chunks],
            "vectors": [self._vectors[cid] for cid in self._chunks],
        }

    def save_json(self, path: str) -> str:
        """Persist as JSON. Human-diffable, which matters when debugging."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False)
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VectorStore":
        if int(payload.get("format", 0)) != cls.FORMAT_VERSION:
            raise ValueError(
                "index format %r is not supported by this version (%d); rebuild the index"
                % (payload.get("format"), cls.FORMAT_VERSION)
            )
        store = cls(int(payload["dim"]), embedder_name=str(payload.get("embedder", "hashing")))
        chunks = [Chunk.from_dict(c) for c in payload["chunks"]]
        store.add(chunks, payload["vectors"])
        return store

    @classmethod
    def load_json(cls, path: str) -> "VectorStore":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @staticmethod
    def npz_available() -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("numpy") is not None
        except Exception:
            return False

    def save_npz(self, path: str) -> str:
        """Persist vectors as a numpy ``.npz`` plus a JSON sidecar of metadata.

        Optional. Worth it above roughly 50k chunks, where JSON float parsing
        starts to dominate startup time.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("numpy is required for .npz persistence") from exc
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        ids = list(self._chunks)
        matrix = np.asarray([self._vectors[cid] for cid in ids], dtype="float32")
        meta = {
            "format": self.FORMAT_VERSION,
            "dim": self.dim,
            "embedder": self.embedder_name,
            "chunks": [self._chunks[cid].to_dict() for cid in ids],
        }
        np.savez_compressed(path, vectors=matrix, meta=json.dumps(meta))
        return path

    @classmethod
    def load_npz(cls, path: str) -> "VectorStore":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("numpy is required for .npz persistence") from exc
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            matrix = data["vectors"]
            store = cls(int(meta["dim"]), embedder_name=str(meta.get("embedder", "hashing")))
            chunks = [Chunk.from_dict(c) for c in meta["chunks"]]
            store.add(chunks, [[float(v) for v in row] for row in matrix])
        return store


class BM25Index:
    """Okapi BM25 over the same tokeniser the rest of the pipeline uses.

    Scoring, written out so the tests can check it by hand::

        idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        score(q, d) = sum_t idf(t) * f(t,d) * (k1 + 1)
                          / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    where ``N`` is the number of indexed chunks, ``df(t)`` the number of chunks
    containing ``t``, ``f(t,d)`` the term frequency in chunk ``d``, ``|d|`` the
    chunk length in tokens and ``avgdl`` the mean chunk length.

    The ``+1`` inside the log is the standard non-negative variant: without it,
    a term appearing in more than half the corpus scores negative and actively
    pushes documents down, which produces baffling rankings on small corpora
    where every chunk mentions the product name.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, *, drop_stopwords: bool = True) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.drop_stopwords = drop_stopwords
        self._chunks: Dict[str, Chunk] = {}
        self._tf: Dict[str, Counter] = {}
        self._len: Dict[str, int] = {}
        self._df: Counter = Counter()
        self._by_doc: Dict[str, List[str]] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def avgdl(self) -> float:
        if not self._len:
            return 0.0
        return sum(self._len.values()) / float(len(self._len))

    def _tokens(self, text: str) -> List[str]:
        return tokenize(text, drop_stopwords=self.drop_stopwords)

    def add(self, chunks: Sequence[Chunk]) -> int:
        added = 0
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                self._remove_one(chunk.chunk_id)
            else:
                added += 1
            toks = self._tokens(chunk.text)
            tf = Counter(toks)
            self._chunks[chunk.chunk_id] = chunk
            self._tf[chunk.chunk_id] = tf
            self._len[chunk.chunk_id] = len(toks)
            for term in tf:
                self._df[term] += 1
            self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)
        return added

    def _remove_one(self, chunk_id: str) -> None:
        tf = self._tf.pop(chunk_id, None)
        if tf is None:
            return
        for term in tf:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._chunks.pop(chunk_id, None)
        self._len.pop(chunk_id, None)

    def delete_document(self, doc_id: str) -> int:
        ids = self._by_doc.pop(doc_id, [])
        for cid in ids:
            self._remove_one(cid)
        return len(ids)

    def upsert_document(self, doc_id: str, chunks: Sequence[Chunk]) -> Tuple[int, int]:
        removed = self.delete_document(doc_id)
        return removed, self.add(chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._tf.clear()
        self._len.clear()
        self._df.clear()
        self._by_doc.clear()

    def idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the non-negative ``ln(1 + x)`` form."""
        n = len(self._chunks)
        if n == 0:
            return 0.0
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, chunk_id: str) -> float:
        """BM25 score of one chunk for one query. Useful for hand-checking."""
        tf = self._tf.get(chunk_id)
        if tf is None:
            return 0.0
        avgdl = self.avgdl or 1.0
        dl = self._len.get(chunk_id, 0)
        denom_len = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
        total = 0.0
        for term in self._tokens(query):
            f = tf.get(term, 0)
            if not f:
                continue
            total += self.idf(term) * (f * (self.k1 + 1.0)) / (f + denom_len)
        return total

    def search(
        self, query: str, k: int = 5, *, where: Optional[MetadataFilter] = None
    ) -> List[ScoredChunk]:
        """BM25 top-k. Chunks scoring exactly zero are not returned."""
        query_terms = self._tokens(query)
        if not query_terms or not self._chunks:
            return []
        avgdl = self.avgdl or 1.0
        idfs = {t: self.idf(t) for t in set(query_terms)}
        scored: List[Tuple[float, str]] = []
        for cid, chunk in self._chunks.items():
            if not _matches(chunk.meta, where):
                continue
            tf = self._tf[cid]
            dl = self._len[cid]
            denom_len = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            total = 0.0
            for term in query_terms:
                f = tf.get(term, 0)
                if f:
                    total += idfs[term] * (f * (self.k1 + 1.0)) / (f + denom_len)
            if total > 0.0:
                scored.append((total, cid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        out: List[ScoredChunk] = []
        for rank, (score, cid) in enumerate(scored[: max(0, k)], start=1):
            out.append(
                ScoredChunk(
                    chunk=self._chunks[cid],
                    score=float(score),
                    rank=rank,
                    components={"bm25": float(score)},
                )
            )
        return out


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[ScoredChunk]:
    """Fuse ranked lists by Reciprocal Rank Fusion.

    ``rrf(c) = sum_i w_i / (k + rank_i(c))`` over the lists that contain ``c``.

    RRF works on *ranks*, not scores, which is the point. A cosine similarity
    of 0.71 and a BM25 score of 8.3 are not comparable, and any attempt to
    normalise them into a weighted sum needs re-tuning every time the corpus
    changes. Ranks need no calibration. The constant ``k`` (60 is the value
    from the original paper and a fine default) damps the influence of the very
    top ranks so that one retriever cannot unilaterally decide the winner.

    A chunk found by *both* retrievers accumulates two terms and therefore
    outranks a chunk that only one retriever found at a similar rank. That is
    the behaviour that makes hybrid retrieval better than either half.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match number of ranked lists")

    fused: Dict[str, float] = {}
    chunks: Dict[str, Chunk] = {}
    components: Dict[str, Dict[str, float]] = {}
    for list_i, (results, weight) in enumerate(zip(ranked_lists, weights)):
        for rank, sc in enumerate(results, start=1):
            cid = sc.chunk.chunk_id
            contribution = weight / (k + rank)
            fused[cid] = fused.get(cid, 0.0) + contribution
            chunks[cid] = sc.chunk
            comp = components.setdefault(cid, {})
            comp["list%d_rank" % list_i] = float(rank)
            comp["list%d_score" % list_i] = float(sc.score)
            comp.update(sc.components)
    order = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    out: List[ScoredChunk] = []
    for rank, (cid, score) in enumerate(order, start=1):
        comp = components[cid]
        comp["rrf"] = float(score)
        comp["retrievers"] = float(sum(1 for key in comp if key.endswith("_rank")))
        out.append(ScoredChunk(chunk=chunks[cid], score=float(score), rank=rank, components=comp))
    return out


@dataclass
class HybridRetriever:
    """Dense + BM25 retrieval fused with RRF.

    Args:
        store: Dense index.
        bm25: Keyword index over the same chunks.
        embedder: Used to embed the query. Must be the embedder the store was
            built with; the store records its name so a mismatch is detectable.
        k_dense / k_sparse: Depth taken from each retriever before fusion. Take
            more than you return: fusion can only reorder what it is given.
        rrf_k: RRF damping constant.
        weights: ``(dense_weight, sparse_weight)``.
    """

    store: VectorStore
    bm25: BM25Index
    embedder: Embedder
    k_dense: int = 20
    k_sparse: int = 20
    rrf_k: int = 60
    weights: Tuple[float, float] = (1.0, 1.0)

    def retrieve(
        self, query: str, k: int = 5, *, where: Optional[MetadataFilter] = None
    ) -> List[ScoredChunk]:
        if not query.strip():
            return []
        dense: List[ScoredChunk] = []
        if len(self.store):
            qvec = self.embedder.embed_one(query)
            dense = self.store.search(qvec, self.k_dense, where=where)
        sparse = self.bm25.search(query, self.k_sparse, where=where) if len(self.bm25) else []
        fused = reciprocal_rank_fusion([dense, sparse], k=self.rrf_k, weights=list(self.weights))
        dense_by_id = {sc.chunk.chunk_id: sc.score for sc in dense}
        sparse_by_id = {sc.chunk.chunk_id: sc.score for sc in sparse}
        for sc in fused:
            sc.components["dense"] = dense_by_id.get(sc.chunk.chunk_id, 0.0)
            sc.components["bm25"] = sparse_by_id.get(sc.chunk.chunk_id, 0.0)
        return fused[: max(0, k)]

    def describe(self) -> Dict[str, Any]:
        return {
            "dense_chunks": len(self.store),
            "sparse_chunks": len(self.bm25),
            "embedder": self.embedder.name,
            "rrf_k": self.rrf_k,
            "weights": list(self.weights),
        }
