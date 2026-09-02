"""Reranking: cheap features that fix the last few positions.

Retrieval gets the right chunk into the top 20. Reranking gets it to position
one. That difference is what the user experiences, because an extractive
answerer (and any sensible LLM prompt) reads the first two or three chunks and
weights them heavily.

:class:`FeatureReranker` is a linear model over five hand-picked features. It
has no learned weights and no model file, so it runs in microseconds and its
decisions are inspectable — every result carries the per-feature contribution
that produced its score. When a customer asks "why did the bot cite the wrong
page", that breakdown is the answer.

A learned cross-encoder is strictly better at semantics and strictly worse at
everything else (latency, dependencies, explainability). It is available as
:class:`CrossEncoderReranker` behind a guarded import.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .textutil import (
    STOPWORDS,
    coverage,
    light_stem,
    longest_common_phrase,
    stem_tokens,
    tokenize,
)
from .types import Chunk, ScoredChunk

__all__ = [
    "Reranker",
    "IdentityReranker",
    "FeatureReranker",
    "CrossEncoderReranker",
    "RerankWeights",
    "compute_match_quality",
    "get_reranker",
    "RERANKER_REGISTRY",
]


@dataclass
class RerankWeights:
    """Linear weights over the reranking features.

    These defaults were tuned against the bundled goldset with
    :mod:`faqbot.eval`, not guessed. They are a starting point for your corpus,
    not a universal setting: ``rare_term`` outweighing ``retrieval`` suits a
    corpus full of part numbers and error codes, and would be the wrong trade
    on a corpus of long prose policy documents.

    ``question_shape`` defaults to **zero**, and that is a measurement, not an
    oversight. The idea is sound — a Q/A block should answer a question better
    than a prose paragraph does — but on a corpus where nearly every chunk is a
    Q/A block it is a near-constant bonus that drowns the rare-term signal. It
    cost 0.05 recall@1 and 0.10 content accuracy on the bundled goldset, so it
    ships switched off. On a corpus that mixes long prose with a small FAQ
    section, turn it back up and re-measure.
    """

    retrieval: float = 0.5
    lexical: float = 0.6
    rare_term: float = 1.3
    breadcrumb: float = 0.5
    exact_phrase: float = 0.7
    position: float = 0.12
    question_shape: float = 0.0
    length_penalty: float = 0.2

    def as_dict(self) -> Dict[str, float]:
        return {
            "retrieval": self.retrieval,
            "lexical": self.lexical,
            "rare_term": self.rare_term,
            "breadcrumb": self.breadcrumb,
            "exact_phrase": self.exact_phrase,
            "position": self.position,
            "question_shape": self.question_shape,
            "length_penalty": self.length_penalty,
        }


def compute_match_quality(
    query: str,
    chunk: "Chunk",
    *,
    idf: Optional[Callable[[str], float]] = None,
) -> float:
    """Absolute, un-normalised match quality of a chunk for a query, in [0, 1].

    Separate from the reranker's own score on purpose. The rerank score mixes
    in a within-candidate-set retrieval prior, so it says "best of what we
    found" rather than "good". The refusal guardrail needs the second question
    answered, and it needs an answer even when reranking is switched off.
    """
    q_tokens = [t for t in tokenize(query) if t not in STOPWORDS]
    c_tokens = tokenize(chunk.text)
    lexical = coverage(q_tokens, c_tokens)
    breadcrumb_text = " ".join(chunk.breadcrumb)
    breadcrumb = coverage(q_tokens, tokenize(breadcrumb_text)) if breadcrumb_text else 0.0
    phrase_len, _ = longest_common_phrase(query, chunk.text, min_tokens=2)
    exact_phrase = min(1.0, math.log1p(max(0, phrase_len - 1)) / math.log1p(6.0))
    if idf is None:
        rare_term = lexical
    else:
        present = set(c_tokens) | stem_tokens(c_tokens)
        total = 0.0
        matched = 0.0
        for term in dict.fromkeys(q_tokens):
            weight = max(0.0, idf(term))
            total += weight
            if term in present or light_stem(term) in present:
                matched += weight
        rare_term = (matched / total) if total > 0 else 0.0
    return 0.35 * lexical + 0.35 * rare_term + 0.15 * breadcrumb + 0.15 * exact_phrase


class Reranker(ABC):
    """Reorder retrieved chunks for one query."""

    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, results: Sequence[ScoredChunk], k: int = 5) -> List[ScoredChunk]:
        """Return at most ``k`` results, best first."""

    @classmethod
    def is_available(cls) -> bool:
        return True


class IdentityReranker(Reranker):
    """Pass-through. Useful as an eval baseline: does reranking help at all?"""

    name = "identity"

    def rerank(self, query: str, results: Sequence[ScoredChunk], k: int = 5) -> List[ScoredChunk]:
        out = list(results)[: max(0, k)]
        for rank, sc in enumerate(out, start=1):
            sc.rank = rank
        return out


class FeatureReranker(Reranker):
    """Explainable linear reranker over five lexical/structural features.

    Features, and the failure each one repairs:

    ``lexical``
        Fraction of the question's content words present in the chunk. Repairs
        the case where a semantically "close" chunk shares topic but not the
        specific thing asked about.

    ``rare_term``
        The same overlap, weighted by corpus IDF. This is the feature that
        separates "the page about error codes" from "the page about error code
        E03", and it needs the keyword index's term statistics, which the
        pipeline wires in.

    ``breadcrumb``
        Overlap between the question and the chunk's heading path. A chunk
        under the heading "How long does the battery last?" is a better answer
        to a battery-life question than a chunk that merely mentions battery in
        passing under "Warranty".

    ``exact_phrase``
        Longest contiguous shared token run, length-damped. This is the feature
        that pulls ``NW-FILT-02`` and "eco mode" to the top; bag-of-words
        similarity treats a matched rare phrase the same as two matched common
        words, and it is not the same.

    ``position``
        Small bonus for chunks earlier in their document. Documentation puts
        the direct answer near the top of a section and the caveats below it.

    ``question_shape``
        Bonus when the chunk is a Q/A block and the query is a question. FAQ
        pairs are answer-shaped; prose paragraphs are not.

    ``length_penalty``
        Mild penalty for very long chunks. A long chunk matches more queries by
        accident, and it dilutes the extractive answerer's sentence scoring.
    """

    name = "feature"

    def __init__(
        self,
        weights: Optional[RerankWeights] = None,
        *,
        target_tokens: int = 180,
        idf: Optional[Callable[[str], float]] = None,
    ) -> None:
        """Args:
            weights: Linear weights over the features.
            target_tokens: Chunk size above which the length penalty starts.
            idf: Corpus IDF lookup, normally ``BM25Index.idf``. Without it the
                ``rare_term`` feature degrades to plain coverage.
        """
        self.weights = weights or RerankWeights()
        self.target_tokens = max(1, target_tokens)
        self.idf = idf

    def _rare_term(self, q_tokens: Sequence[str], c_tokens: Sequence[str]) -> float:
        """IDF-weighted share of the question's terms that the chunk matches.

        Plain coverage treats every query word alike, so matching ``error`` and
        ``code`` scores the same as matching ``E03``. On a support corpus that
        is backwards: the rare identifier *is* the question. Weighting each
        matched term by its corpus IDF makes an exact hit on ``E03``,
        ``NW-FILT-02`` or ``NW-BATT-32`` worth more than several common words,
        which is the single change that moves the right error-code page to
        rank one.
        """
        if not q_tokens:
            return 0.0
        idf = self.idf
        if idf is None:
            return coverage(q_tokens, c_tokens)
        present = set(c_tokens) | stem_tokens(c_tokens)
        total = 0.0
        matched = 0.0
        for term in dict.fromkeys(q_tokens):
            weight = max(0.0, idf(term))
            total += weight
            if term in present or light_stem(term) in present:
                matched += weight
        return (matched / total) if total > 0 else 0.0

    def features(self, query: str, sc: ScoredChunk) -> Dict[str, float]:
        chunk = sc.chunk
        q_tokens = [t for t in tokenize(query) if t not in STOPWORDS]
        c_tokens = tokenize(chunk.text)
        lexical = coverage(q_tokens, c_tokens)
        rare_term = self._rare_term(q_tokens, c_tokens)

        breadcrumb_text = " ".join(chunk.breadcrumb)
        breadcrumb = coverage(q_tokens, tokenize(breadcrumb_text)) if breadcrumb_text else 0.0

        phrase_len, _ = longest_common_phrase(query, chunk.text, min_tokens=2)
        # Damped: going from a 2-token to a 4-token match matters much more
        # than going from 8 to 10.
        exact_phrase = math.log1p(max(0, phrase_len - 1)) / math.log1p(6.0)
        exact_phrase = min(1.0, exact_phrase)

        position = 1.0 / (1.0 + float(chunk.index))

        is_question_chunk = bool(chunk.meta.get("is_question")) or chunk.meta.get("kind") == "faq_pair"
        query_is_question = query.strip().endswith("?") or bool(q_tokens and q_tokens[0] in {
            "how", "what", "why", "when", "where", "who", "which", "can", "does", "do", "is", "are",
        })
        question_shape = 1.0 if (is_question_chunk and query_is_question) else 0.0

        n_tokens = len(c_tokens)
        length_penalty = max(0.0, (n_tokens - self.target_tokens) / float(self.target_tokens))
        length_penalty = min(1.0, length_penalty)

        return {
            "retrieval": float(sc.score),
            "lexical": lexical,
            "rare_term": rare_term,
            "breadcrumb": breadcrumb,
            "exact_phrase": exact_phrase,
            "position": position,
            "question_shape": question_shape,
            "length_penalty": length_penalty,
        }

    def rerank(self, query: str, results: Sequence[ScoredChunk], k: int = 5) -> List[ScoredChunk]:
        if not results:
            return []
        weights = self.weights.as_dict()

        rescored: List[ScoredChunk] = []
        for rank0, sc in enumerate(results):
            feats = self.features(query, sc)
            # The retrieval prior is the candidate's *rank*, not its score.
            # RRF values and cosine magnitudes are not comparable across
            # queries, and min-max normalising them inflates a meaningless gap
            # (0.0164 vs 0.0315 RRF) into a full point of score. Reciprocal
            # rank decays smoothly and cannot be gamed by scale.
            feats["retrieval"] = 1.0 / (1.0 + rank0 / 4.0)
            score = 0.0
            contributions: Dict[str, float] = {}
            for key, weight in weights.items():
                value = feats.get(key, 0.0)
                contribution = -weight * value if key == "length_penalty" else weight * value
                contributions["f_" + key] = round(value, 4)
                contributions["w_" + key] = round(contribution, 4)
                score += contribution
            # An absolute, un-normalised measure of how well this chunk
            # matches the question, in [0, 1]. The final rerank score is only
            # meaningful *within* one candidate set (its retrieval term is
            # min-max normalised), so it cannot be compared against a fixed
            # confidence threshold. match_quality can, and that is what the
            # refusal guardrail reads.
            match_quality = (
                0.35 * feats["lexical"]
                + 0.35 * feats["rare_term"]
                + 0.15 * feats["breadcrumb"]
                + 0.15 * feats["exact_phrase"]
            )
            merged = dict(sc.components)
            merged.update(contributions)
            merged["rerank"] = score
            merged["match_quality"] = round(match_quality, 4)
            merged["pre_rerank_score"] = float(sc.score)
            rescored.append(ScoredChunk(chunk=sc.chunk, score=score, components=merged))

        rescored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        out = rescored[: max(0, k)]
        for rank, sc in enumerate(out, start=1):
            sc.rank = rank
        return out


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranking via ``sentence-transformers``. Guarded import.

    A cross-encoder reads the query and the chunk *together* and scores the
    pair, which is why it beats any bag-of-words feature set on paraphrase. The
    cost is a forward pass per candidate: reranking 50 candidates is 50
    inferences. Retrieve wide, rerank narrow.
    """

    name = "cross-encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        fallback: Optional[Reranker] = None,
    ) -> None:
        self.model_name = model_name
        self.fallback = fallback or FeatureReranker()
        self._model: Any = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            return False

    def rerank(self, query: str, results: Sequence[ScoredChunk], k: int = 5) -> List[ScoredChunk]:
        if not self.is_available():
            # Degrade, do not crash. A missing optional model must never take
            # the service down; it should quietly cost a little accuracy.
            return self.fallback.rerank(query, results, k)
        return self._rerank_model(query, results, k)

    def _rerank_model(
        self, query: str, results: Sequence[ScoredChunk], k: int
    ) -> List[ScoredChunk]:  # pragma: no cover - optional dep
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        pairs = [(query, sc.chunk.text) for sc in results]
        scores = self._model.predict(pairs)
        rescored = [
            ScoredChunk(
                chunk=sc.chunk,
                score=float(score),
                components={**sc.components, "cross_encoder": float(score)},
            )
            for sc, score in zip(results, scores)
        ]
        rescored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        out = rescored[: max(0, k)]
        for rank, sc in enumerate(out, start=1):
            sc.rank = rank
        return out


RERANKER_REGISTRY: Dict[str, type] = {
    "identity": IdentityReranker,
    "feature": FeatureReranker,
    "cross-encoder": CrossEncoderReranker,
}


def get_reranker(name: str = "feature", **kwargs: Any) -> Reranker:
    """Build a reranker by name."""
    try:
        cls = RERANKER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            "unknown reranker %r; available: %s" % (name, ", ".join(sorted(RERANKER_REGISTRY)))
        ) from None
    return cls(**kwargs)  # type: ignore[arg-type]
