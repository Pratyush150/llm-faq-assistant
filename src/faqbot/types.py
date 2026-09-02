"""Core data structures shared by every stage of the pipeline.

Everything that moves between stages is a frozen-ish dataclass with explicit
fields. There is no dict-of-unknown-shape passed around, because the single
hardest thing to debug in a retrieval system is "which stage dropped my
metadata".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "Document",
    "Chunk",
    "ScoredChunk",
    "Citation",
    "Answer",
    "RefusalReason",
]


@dataclass
class Document:
    """A normalised source document.

    Attributes:
        doc_id: Content hash. Two ingests of identical content produce the same
            id, which is what makes re-ingestion idempotent.
        source: Where it came from (file path, URL, or a synthetic label).
        title: Best-effort human title (first heading, CSV question, filename).
        text: Normalised plain text or markdown.
        meta: Free-form metadata used for filtering at retrieval time.
    """

    doc_id: str
    source: str
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        return cls(
            doc_id=d["doc_id"],
            source=d["source"],
            title=d.get("title", ""),
            text=d["text"],
            meta=dict(d.get("meta", {})),
        )


@dataclass
class Chunk:
    """A retrievable unit of text.

    ``breadcrumb`` is the heading path the chunk was found under, e.g.
    ``("Battery & charging", "How long does a charge take?")``. It is not
    decoration: the reranker scores it, the answerer shows it as a citation
    label, and it is the cheapest way to give a chunk the context that
    splitting stripped from it.
    """

    chunk_id: str
    doc_id: str
    text: str
    index: int = 0
    breadcrumb: Tuple[str, ...] = ()
    start_char: int = 0
    end_char: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.meta.get("source", ""))

    @property
    def label(self) -> str:
        """Short human label used in citations."""
        if self.breadcrumb:
            return " > ".join(self.breadcrumb)
        return self.meta.get("title") or self.source or self.chunk_id

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["breadcrumb"] = list(self.breadcrumb)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            text=d["text"],
            index=int(d.get("index", 0)),
            breadcrumb=tuple(d.get("breadcrumb", ())),
            start_char=int(d.get("start_char", 0)),
            end_char=int(d.get("end_char", 0)),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class ScoredChunk:
    """A chunk plus the score that retrieved it, plus how it got that score.

    ``components`` keeps the per-signal breakdown (dense, sparse, rrf, rerank
    features). Without it, "why did this chunk come back?" is unanswerable and
    tuning becomes superstition.
    """

    chunk: Chunk
    score: float
    rank: int = 0
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "components": dict(self.components),
        }


@dataclass
class Citation:
    """A pointer from one answer sentence back into the corpus."""

    chunk_id: str
    doc_id: str
    source: str
    label: str
    quote: str
    start_char: int
    end_char: int
    score: float = 0.0
    marker: int = 1
    """Footnote number shown in the answer text. Sentences quoted from the same
    chunk share a marker, so an answer built from one page shows ``[1]`` three
    times rather than ``[1] [2] [3]``."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RefusalReason:
    """Stable string constants for why an answer was withheld."""

    NONE = ""
    LOW_CONFIDENCE = "low_retrieval_confidence"
    OUT_OF_DOMAIN = "out_of_domain"
    AMBIGUOUS = "ambiguous_question"
    CONTRADICTORY = "contradictory_sources"
    UNGROUNDED = "answer_not_grounded"
    NO_CONTEXT = "no_context_retrieved"
    INJECTION = "retrieved_content_injection"


@dataclass
class Answer:
    """The single return type of every answerer, extractive or LLM-backed.

    ``refused`` is a first-class field, not an exception and not a magic
    string in ``text``. A caller that wants to route unanswered questions to a
    human can branch on one boolean.
    """

    text: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    refusal_reason: str = RefusalReason.NONE
    question: str = ""
    rewritten_question: str = ""
    groundedness: float = 0.0
    latency_ms: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> List[str]:
        """Ids of every chunk this answer cites, in citation order."""
        seen: List[str] = []
        for c in self.citations:
            if c.chunk_id not in seen:
                seen.append(c.chunk_id)
        return seen

    @property
    def doc_ids(self) -> List[str]:
        seen: List[str] = []
        for c in self.citations:
            if c.doc_id not in seen:
                seen.append(c.doc_id)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "chunk_ids": self.chunk_ids,
            "doc_ids": self.doc_ids,
            "confidence": round(self.confidence, 4),
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "groundedness": round(self.groundedness, 4),
            "latency_ms": round(self.latency_ms, 2),
            "diagnostics": self.diagnostics,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def render(self, width: int = 78) -> str:
        """Plain-text rendering used by the CLI."""
        lines: List[str] = []
        lines.append(_wrap(self.text, width))
        if self.rewritten_question and self.rewritten_question != self.question:
            lines.append("")
            lines.append("  (resolved question: %s)" % self.rewritten_question)
        if self.citations:
            lines.append("")
            lines.append("sources:")
            shown: List[int] = []
            for c in self.citations:
                if c.marker in shown:
                    continue
                shown.append(c.marker)
                lines.append("  [%d] %s  (%s)" % (c.marker, c.label, c.source or c.doc_id))
        lines.append("")
        status = "REFUSED (%s)" % self.refusal_reason if self.refused else "answered"
        lines.append(
            "  %s | confidence %.2f | groundedness %.2f | %.1f ms"
            % (status, self.confidence, self.groundedness, self.latency_ms)
        )
        return "\n".join(lines)


def _wrap(text: str, width: int) -> str:
    import textwrap

    out: List[str] = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(out)
