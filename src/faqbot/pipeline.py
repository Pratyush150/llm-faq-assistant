"""The pipeline: ingest -> chunk -> embed -> store -> retrieve -> rerank ->
guardrails -> answer, plus conversation memory and query rewriting.

One dataclass configures the whole thing, so an A/B comparison of two chunking
strategies is two :class:`PipelineConfig` values and a loop, not two branches of
a codebase. :mod:`faqbot.eval` relies on exactly that.

Conversation memory exists because real support conversations are elliptical.
The second question is almost never standalone::

    user: How long does the AR-1 battery last?
    bot:  Up to 90 minutes in Eco mode ...
    user: Does it work on carpet?

Embedding "Does it work on carpet?" retrieves nothing useful: the words that
identify the subject are in the *previous* turn. :class:`QueryRewriter` rewrites
it to "Does the AR-1 work on carpet?" before retrieval. The rewrite is
deterministic and rule-based rather than a model call, so it costs nothing, it
is testable, and it is visible in the response as ``rewritten_question`` — the
user can see what the bot thought they meant.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from .answer import Answerer, ExtractiveAnswerer
from .chunking import Chunker, get_chunker
from .embedding import Embedder, get_embedder
from .guardrails import GuardrailConfig, Guardrails, neutralize_retrieved_text
from .ingest import IngestConfig, load_paths
from .rerank import Reranker, compute_match_quality, get_reranker
from .store import BM25Index, HybridRetriever, VectorStore
from .textutil import STOPWORDS, tokenize
from .types import Answer, Chunk, Document, RefusalReason, ScoredChunk

__all__ = [
    "PipelineConfig",
    "FAQPipeline",
    "ConversationMemory",
    "QueryRewriter",
    "Turn",
]


# --------------------------------------------------------------------------
# Conversation memory and query rewriting
# --------------------------------------------------------------------------


@dataclass
class Turn:
    """One question/answer exchange."""

    question: str
    rewritten: str
    answer: str
    entities: Tuple[str, ...] = ()
    refused: bool = False


_ENTITY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*(?:\s+(?:[A-Z][A-Za-z0-9]*|[0-9]+))*)\b"
)
_PRONOUNS = {"it", "its", "that", "this", "they", "them", "those", "these", "one", "ones"}
_FOLLOWUP_STARTS = (
    "what about", "how about", "and what about", "what if", "and ", "but ", "also ",
    "ok what about", "okay what about",
)
_STOP_ENTITIES = {
    "how", "what", "why", "when", "where", "who", "which", "can", "does", "do",
    "is", "are", "will", "should", "the", "a", "an", "i", "my", "we",
}


def extract_entities(text: str) -> List[str]:
    """Pull product-ish noun phrases out of a question or answer.

    Targets capitalised words, model numbers and part codes (``AR-1``,
    ``AR-1 Pro``, ``NW-FILT-02``) because those are the subjects a follow-up
    question drops. Deliberately shallow: a POS tagger would be more accurate
    and would add a dependency, a model download and a startup cost to solve
    a problem that a support corpus mostly does not have.
    """
    found: List[str] = []
    for m in _ENTITY_RE.finditer(text):
        phrase = m.group(1).strip()
        if phrase.casefold() in _STOP_ENTITIES:
            continue
        if len(phrase) < 2:
            continue
        # Require either a digit/hyphen (model number) or a non-initial position.
        if not (any(ch.isdigit() for ch in phrase) or "-" in phrase or m.start() > 0):
            continue
        if phrase not in found:
            found.append(phrase)
    return found


class ConversationMemory:
    """Bounded per-session history.

    Bounded on purpose. Unbounded history is how a chatbot session turns into
    an ever-growing context that costs more every turn and eventually drags in
    a topic from twenty questions ago.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max(1, max_turns)
        self._turns: Deque[Turn] = deque(maxlen=self.max_turns)

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)

    def add(self, turn: Turn) -> None:
        self._turns.append(turn)

    def last(self) -> Optional[Turn]:
        return self._turns[-1] if self._turns else None

    def salient_entity(self) -> Optional[str]:
        """The most recent named subject, searching backwards.

        Questions first, then their answers: the user's own wording is a better
        subject than anything the bot echoed back.
        """
        for turn in reversed(self._turns):
            if turn.entities:
                return turn.entities[0]
        for turn in reversed(self._turns):
            ents = extract_entities(turn.answer)
            if ents:
                return ents[0]
        return None

    def clear(self) -> None:
        self._turns.clear()


class QueryRewriter:
    """Rule-based follow-up resolution. Deterministic, testable, free.

    Three rewrite forms, tried in order:

    1. **Pronoun substitution** — a dangling ``it``/``that``/``they`` is
       replaced with the salient entity from memory.
    2. **Topic switch** — ``what about the Pro version?`` re-asks the previous
       question with the new subject substituted in.
    3. **Bare fragment** — a two-word query with no verb inherits the previous
       question's predicate.

    If none apply, the query is returned untouched. Rewriting a question that
    did not need it is worse than not rewriting: it drags the previous topic
    into an unrelated question and retrieves the wrong page.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    @staticmethod
    def _needs_rewrite(question: str) -> bool:
        low = question.strip().casefold()
        if any(low.startswith(p) for p in _FOLLOWUP_STARTS):
            return True
        toks = tokenize(question)
        if any(t in _PRONOUNS for t in toks) and not extract_entities(question):
            return True
        content = [t for t in toks if t not in STOPWORDS]
        return len(content) <= 2 and not extract_entities(question)

    @staticmethod
    def _article(entity: str) -> str:
        return entity if entity.casefold().startswith(("the ", "a ", "an ")) else "the " + entity

    def rewrite(
        self, question: str, memory: Optional[ConversationMemory]
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """Return ``(rewritten, changed, info)``."""
        info: Dict[str, Any] = {"rule": "none"}
        if not self.enabled or memory is None or len(memory) == 0:
            return question, False, info
        if not self._needs_rewrite(question):
            return question, False, info

        last = memory.last()
        entity = memory.salient_entity()
        if last is None:
            return question, False, info

        low = question.strip().casefold()

        # 2. Topic switch: "what about the Pro version?"
        for prefix in ("what about", "how about", "and what about", "ok what about", "okay what about"):
            if low.startswith(prefix):
                new_subject = question.strip()[len(prefix) :].strip(" ?.!,:")
                if new_subject and entity:
                    base = last.rewritten or last.question
                    pattern = re.compile(r"(the\s+)?" + re.escape(entity), re.IGNORECASE)
                    replacement = new_subject if new_subject.casefold().startswith(
                        ("the ", "a ", "an ")
                    ) else "the " + new_subject
                    rewritten, n = pattern.subn(replacement, base, count=1)
                    if n:
                        info = {"rule": "topic_switch", "entity": entity, "new_subject": new_subject}
                        return rewritten, True, info
                if new_subject:
                    info = {"rule": "topic_switch_append", "new_subject": new_subject}
                    return "%s %s" % (last.rewritten or last.question, new_subject), True, info

        # 1. Pronoun substitution.
        if entity:
            toks = tokenize(question)
            if any(t in _PRONOUNS for t in toks):
                pattern = re.compile(
                    r"\b(%s)\b" % "|".join(sorted(_PRONOUNS, key=len, reverse=True)),
                    re.IGNORECASE,
                )
                rewritten, n = pattern.subn(self._article(entity), question, count=1)
                if n:
                    info = {"rule": "pronoun", "entity": entity}
                    return rewritten, True, info

            # 3. Bare fragment: "battery life?" after a question about the AR-1.
            content = [t for t in tokenize(question) if t not in STOPWORDS]
            if len(content) <= 2:
                info = {"rule": "fragment", "entity": entity}
                return "%s %s" % (question.strip().rstrip("?").strip(), self._article(entity)), True, info

        return question, False, info


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Every knob in one place.

    Two configurations differing only in ``chunker`` are directly comparable in
    the eval harness, which is the whole reason this is a dataclass and not a
    pile of constructor arguments.
    """

    # ingest
    ingest: IngestConfig = field(default_factory=IngestConfig)

    # chunking
    chunker: str = "structure"
    chunker_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_tokens": 180, "overlap_sentences": 1})

    # embedding
    embedder: str = "hashing"
    embedder_kwargs: Dict[str, Any] = field(default_factory=lambda: {"dim": 256})

    # retrieval
    top_k: int = 5
    k_dense: int = 20
    k_sparse: int = 20
    rrf_k: int = 60
    hybrid_weights: Tuple[float, float] = (1.0, 1.0)
    use_dense: bool = True
    use_sparse: bool = True

    # reranking
    reranker: str = "feature"
    reranker_kwargs: Dict[str, Any] = field(default_factory=dict)
    rerank_candidates: int = 20

    # answering
    answerer: str = "extractive"
    max_answer_sentences: int = 3

    # guardrails
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    refuse_on_injection: bool = False
    """If False (default) injected content is neutralised and the answer
    proceeds; if True the question is refused outright. Refusing is the safer
    posture for high-stakes corpora and a denial-of-service opportunity for
    anyone who can edit an indexed page, so it is off by default."""

    # conversation
    memory_turns: int = 6
    rewrite_queries: bool = True

    label: str = "default"
    """Name used in eval comparison tables."""

    def describe(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "chunker": self.chunker,
            "chunker_kwargs": dict(self.chunker_kwargs),
            "embedder": self.embedder,
            "top_k": self.top_k,
            "reranker": self.reranker,
            "answerer": self.answerer,
            "dense": self.use_dense,
            "sparse": self.use_sparse,
            "rewrite_queries": self.rewrite_queries,
        }


class FAQPipeline:
    """End-to-end retrieval-grounded question answering.

    Typical use::

        pipe = FAQPipeline()
        pipe.ingest(["data/corpus"])
        print(pipe.ask("How long does the AR-1 battery last?").render())
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
        reranker: Optional[Reranker] = None,
        answerer: Optional[Answerer] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.embedder = embedder or get_embedder(self.config.embedder, **self.config.embedder_kwargs)
        self.chunker = chunker or get_chunker(self.config.chunker, **self.config.chunker_kwargs)
        self.bm25 = BM25Index()
        if reranker is not None:
            self.reranker: Optional[Reranker] = reranker
        elif self.config.reranker == "none":
            self.reranker = None
        else:
            kwargs = dict(self.config.reranker_kwargs)
            if self.config.reranker == "feature":
                # Give the reranker the keyword index's term statistics so its
                # rare-term feature can tell "E03" apart from "error".
                kwargs.setdefault("idf", self.bm25.idf)
            self.reranker = get_reranker(self.config.reranker, **kwargs)
        self.answerer = answerer or ExtractiveAnswerer(max_sentences=self.config.max_answer_sentences)
        self.store = VectorStore(self.embedder.dim, embedder_name=self.embedder.name)
        self.guards = Guardrails(self.config.guardrails)
        self.rewriter = QueryRewriter(enabled=self.config.rewrite_queries)
        self.retriever = HybridRetriever(
            store=self.store,
            bm25=self.bm25,
            embedder=self.embedder,
            k_dense=self.config.k_dense,
            k_sparse=self.config.k_sparse,
            rrf_k=self.config.rrf_k,
            weights=self.config.hybrid_weights,
        )
        self._documents: Dict[str, Document] = {}
        self._sessions: Dict[str, ConversationMemory] = {}

    # -- indexing --------------------------------------------------------
    def add_documents(self, docs: Sequence[Document]) -> Dict[str, int]:
        """Chunk, embed and upsert documents. Idempotent per ``doc_id``.

        Returns counts that distinguish *new* work from *repeated* work.
        Re-running an ingest over an unchanged corpus reports zero chunks added
        and every chunk replaced, which is the signal an operator needs to see
        that a nightly re-crawl is not quietly duplicating the index.
        """
        added_chunks = 0
        replaced = 0
        new_documents = 0
        for doc in docs:
            chunks = self.chunker.split(doc)
            if not chunks:
                continue
            is_new = doc.doc_id not in self._documents
            vectors = self.embedder.embed([c.text for c in chunks])
            removed, written = self.store.upsert_document(doc.doc_id, chunks, vectors)
            self.bm25.upsert_document(doc.doc_id, chunks)
            replaced += removed
            if is_new:
                new_documents += 1
                added_chunks += written
            self._documents[doc.doc_id] = doc
        self.guards.fit(self.store.chunks)
        return {
            "documents": len(docs),
            "documents_new": new_documents,
            "chunks_added": added_chunks,
            "chunks_replaced": replaced,
            "chunks_total": len(self.store),
        }

    def ingest(self, paths: Sequence[str]) -> Dict[str, int]:
        """Load files or directories and index them."""
        docs = load_paths(paths, self.config.ingest)
        stats = self.add_documents(docs)
        stats["paths"] = len(paths)
        return stats

    def delete_document(self, doc_id: str) -> int:
        self._documents.pop(doc_id, None)
        self.bm25.delete_document(doc_id)
        removed = self.store.delete_document(doc_id)
        self.guards.fit(self.store.chunks)
        return removed

    @property
    def documents(self) -> List[Document]:
        return list(self._documents.values())

    def stats(self) -> Dict[str, Any]:
        out = self.store.stats()
        out.update(
            {
                "bm25_chunks": len(self.bm25),
                "vocabulary_terms": len(self.guards.vocab),
                "config": self.config.describe(),
            }
        )
        return out

    # -- retrieval -------------------------------------------------------
    def retrieve(
        self, query: str, k: Optional[int] = None, *, where: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        """Hybrid retrieve, then rerank. Returns at most ``k`` chunks."""
        k = self.config.top_k if k is None else k
        if not len(self.store) and not len(self.bm25):
            return []
        if self.config.use_dense and self.config.use_sparse:
            candidates = self.retriever.retrieve(query, self.config.rerank_candidates, where=where)
        elif self.config.use_dense:
            qvec = self.embedder.embed_one(query)
            candidates = self.store.search(qvec, self.config.rerank_candidates, where=where)
        elif self.config.use_sparse:
            candidates = self.bm25.search(query, self.config.rerank_candidates, where=where)
        else:
            raise ValueError("at least one of use_dense / use_sparse must be True")
        if self.reranker is None:
            out = candidates[:k]
            # The guardrails read match_quality. Without a reranker nothing has
            # computed it, and a missing signal must not turn into "refuse
            # everything": fill it in here so switching reranking off changes
            # ranking only, never the refusal behaviour.
            idf = self.bm25.idf if len(self.bm25) else None
            for sc in out:
                sc.components.setdefault(
                    "match_quality", compute_match_quality(query, sc.chunk, idf=idf)
                )
            return out
        return self.reranker.rerank(query, candidates, k)

    # -- conversation ----------------------------------------------------
    def memory(self, session_id: str = "default") -> ConversationMemory:
        return self._sessions.setdefault(session_id, ConversationMemory(self.config.memory_turns))

    def reset_session(self, session_id: str = "default") -> None:
        self._sessions.pop(session_id, None)

    # -- answering -------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        session_id: Optional[str] = None,
        k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Answer:
        """Answer one question end to end.

        Order matters. Retrieval happens on the rewritten question; the
        injection scan happens before the answerer sees anything; grounding is
        checked after the answer exists. Every stage's output is recorded in
        ``Answer.diagnostics`` so a wrong answer can be traced to the stage
        that caused it rather than blamed on "the model".
        """
        started = time.perf_counter()
        memory = self.memory(session_id) if session_id else None
        rewritten, changed, rewrite_info = self.rewriter.rewrite(question, memory)
        query = rewritten if changed else question

        results = self.retrieve(query, k, where=where)

        injection_hits = self.guards.scan_context([sc.chunk for sc in results])
        if injection_hits and self.config.refuse_on_injection:
            return self._finish(
                Answer(
                    text=(
                        "One of the matching documents contains embedded instructions, "
                        "so I am not answering from it. A human should review that page."
                    ),
                    refused=True,
                    refusal_reason=RefusalReason.INJECTION,
                    question=question,
                    rewritten_question=query,
                    diagnostics={"injection_hits": injection_hits},
                ),
                started,
                memory,
                question,
                query,
            )

        safe_results = _sanitize(results) if injection_hits else results

        verdict = self.guards.pre_answer(query, safe_results, resolved=changed)
        if not verdict.allow:
            return self._finish(
                Answer(
                    text=self.guards.refusal_text(verdict),
                    refused=True,
                    refusal_reason=verdict.reason,
                    question=question,
                    rewritten_question=query,
                    confidence=float(verdict.scores.get("top_score", 0.0)),
                    diagnostics={
                        "guardrail": verdict.to_dict(),
                        "rewrite": rewrite_info,
                        "retrieved": [sc.chunk.chunk_id for sc in safe_results],
                        "injection_hits": injection_hits,
                    },
                ),
                started,
                memory,
                question,
                query,
            )

        answer = self.answerer.answer(query, safe_results)
        answer.question = question
        answer.rewritten_question = query
        answer.diagnostics.setdefault("guardrail", verdict.to_dict())
        answer.diagnostics["rewrite"] = rewrite_info
        answer.diagnostics["injection_hits"] = injection_hits
        answer.diagnostics["retrieved"] = [
            {"chunk_id": sc.chunk.chunk_id, "score": round(sc.score, 4), "label": sc.chunk.label}
            for sc in safe_results
        ]

        cited_texts = [c.quote for c in answer.citations] or [sc.chunk.text for sc in safe_results[:1]]
        post = self.guards.post_answer(answer.text, [self._chunk_text(c.chunk_id, safe_results) for c in answer.citations] or cited_texts)
        answer.groundedness = float(post.scores.get("groundedness", 0.0))
        if not post.allow:
            answer.refused = True
            answer.refusal_reason = post.reason
            answer.text = self.guards.refusal_text(post)
            answer.citations = []
            answer.diagnostics["grounding"] = post.to_dict()

        return self._finish(answer, started, memory, question, query)

    @staticmethod
    def _chunk_text(chunk_id: str, results: Sequence[ScoredChunk]) -> str:
        for sc in results:
            if sc.chunk.chunk_id == chunk_id:
                return sc.chunk.text
        return ""

    def _finish(
        self,
        answer: Answer,
        started: float,
        memory: Optional[ConversationMemory],
        question: str,
        query: str,
    ) -> Answer:
        answer.latency_ms = (time.perf_counter() - started) * 1000.0
        answer.question = question
        answer.rewritten_question = query
        if memory is not None:
            memory.add(
                Turn(
                    question=question,
                    rewritten=query,
                    answer=answer.text,
                    entities=tuple(extract_entities(query) or extract_entities(question)),
                    refused=answer.refused,
                )
            )
        return answer

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> str:
        """Persist the dense index (JSON, or npz when the path ends in .npz)."""
        if path.endswith(".npz"):
            return self.store.save_npz(path)
        return self.store.save_json(path)

    def load(self, path: str) -> "FAQPipeline":
        """Load a persisted index and rebuild the keyword index from it.

        BM25 is rebuilt rather than serialised: it is cheap to recompute and
        keeping a second serialisation format in sync is a reliable source of
        "the two indexes contain different documents" bugs.
        """
        store = VectorStore.load_npz(path) if path.endswith(".npz") else VectorStore.load_json(path)
        if store.dim != self.embedder.dim:
            raise ValueError(
                "index dim %d does not match embedder dim %d (index was built with %r)"
                % (store.dim, self.embedder.dim, store.embedder_name)
            )
        self.store = store
        self.bm25.clear()
        self.bm25.add(store.chunks)
        self.retriever.store = store
        self.retriever.bm25 = self.bm25
        self.guards.fit(store.chunks)
        return self


def _sanitize(results: Sequence[ScoredChunk]) -> List[ScoredChunk]:
    """Return copies of the results with instruction-shaped spans defanged.

    The answerer never sees the raw text of a chunk that tripped the injection
    scan. For the extractive answerer that means an injected imperative cannot
    be quoted back to the user as if it were product documentation; for an LLM
    answerer it means the imperative is not in the prompt at all.
    """
    out: List[ScoredChunk] = []
    for sc in results:
        chunk = sc.chunk
        clean = neutralize_retrieved_text(chunk.text)
        if clean == chunk.text:
            out.append(sc)
            continue
        safe = Chunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=clean,
            index=chunk.index,
            breadcrumb=chunk.breadcrumb,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            meta={**chunk.meta, "neutralised": True},
        )
        out.append(ScoredChunk(chunk=safe, score=sc.score, rank=sc.rank, components=dict(sc.components)))
    return out
