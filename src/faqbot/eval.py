"""Evaluation harness. Without this the rest of the repository is an opinion.

"Our chatbot works" is not a claim anyone can act on. These are:

* **recall@k** — how often the passage that contains the answer is in the top
  k. If this is low, nothing downstream can fix it; the answer is not in the
  context window.
* **MRR** — where in the list the right passage lands. Recall@5 of 0.9 with an
  MRR of 0.3 means the right chunk is usually present and usually buried, which
  is a reranking problem, not a retrieval problem.
* **groundedness** — fraction of answer sentences actually supported by the
  cited text. This is the hallucination rate, inverted.
* **citation precision** — fraction of citations that point at a passage the
  goldset marks relevant. Catches the answer that is right for the wrong
  reason, which is the one that breaks silently later.
* **refusal correctness** — on questions the corpus genuinely cannot answer,
  did it refuse? And on questions it can, did it wrongly refuse? A bot that
  refuses everything scores perfectly on the first and terribly on the second,
  which is why both are reported.
* **latency percentiles** — p50/p90/p95, because a p50 of 40 ms with a p95 of
  four seconds is a support queue full of people who think it is broken.

:func:`compare` runs the same goldset across several
:class:`~faqbot.pipeline.PipelineConfig` values and prints a table, so
"structure-aware chunking beats fixed-size chunking" becomes a number on your
corpus rather than a claim in a README.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pipeline import FAQPipeline, PipelineConfig
from .types import Answer, ScoredChunk

__all__ = [
    "GoldQuestion",
    "GoldSet",
    "load_goldset",
    "QuestionResult",
    "EvalReport",
    "run_eval",
    "compare",
    "format_comparison_table",
    "recall_at_k",
    "mean_reciprocal_rank",
    "percentile",
]


# --------------------------------------------------------------------------
# Goldset
# --------------------------------------------------------------------------


@dataclass
class GoldQuestion:
    """One evaluation item.

    Attributes:
        qid: Stable identifier, used in reports.
        question: What the user types.
        answerable: False for questions the corpus deliberately cannot answer.
            These are the refusal tests, and a goldset without them cannot tell
            a careful system from a confident one.
        relevant_sources: Substrings matched against a chunk's ``source``. Path
            substrings rather than content hashes, so a goldset stays valid
            when the corpus is re-ingested.
        expected_contains: Strings the answer text should contain. Case
            insensitive, all must be present to count as a content hit.
        notes: Free text, shown in per-question failure output.
    """

    qid: str
    question: str
    answerable: bool = True
    relevant_sources: Tuple[str, ...] = ()
    expected_contains: Tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoldQuestion":
        return cls(
            qid=str(d.get("id") or d.get("qid") or ""),
            question=str(d["question"]),
            answerable=bool(d.get("answerable", True)),
            relevant_sources=tuple(d.get("relevant_sources", ()) or ()),
            expected_contains=tuple(d.get("expected_contains", ()) or ()),
            notes=str(d.get("notes", "")),
        )


@dataclass
class GoldSet:
    """A named set of questions plus the corpus they are about."""

    name: str
    corpus: str
    questions: List[GoldQuestion] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    @property
    def answerable(self) -> List[GoldQuestion]:
        return [q for q in self.questions if q.answerable]

    @property
    def unanswerable(self) -> List[GoldQuestion]:
        return [q for q in self.questions if not q.answerable]


def load_goldset(path: str) -> GoldSet:
    """Load a goldset from JSON, or from YAML when PyYAML is installed.

    JSON is the supported baseline precisely so the harness has no hard
    dependency; YAML is nicer to hand-edit and is used when available.
    """
    suffix = os.path.splitext(path)[1].casefold()
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "PyYAML is not installed; convert the goldset to JSON or "
                "`pip install PyYAML`."
            ) from exc
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if isinstance(data, list):
        data = {"questions": data}
    base = os.path.dirname(os.path.abspath(path))
    corpus = str(data.get("corpus", ""))
    if corpus and not os.path.isabs(corpus):
        corpus = os.path.normpath(os.path.join(base, corpus))
    return GoldSet(
        name=str(data.get("name", os.path.basename(path))),
        corpus=corpus,
        questions=[GoldQuestion.from_dict(q) for q in data["questions"]],
    )


# --------------------------------------------------------------------------
# Metric primitives
# --------------------------------------------------------------------------


def _is_relevant(sc: ScoredChunk, relevant_sources: Sequence[str]) -> bool:
    source = sc.chunk.source or str(sc.chunk.meta.get("source", ""))
    return any(marker in source for marker in relevant_sources)


def recall_at_k(hits: Sequence[bool]) -> float:
    """Fraction of queries with at least one relevant result in the top k.

    Takes the per-query booleans rather than raw results so the same function
    serves the harness and a hand-computed unit test.
    """
    if not hits:
        return 0.0
    return sum(1 for h in hits if h) / float(len(hits))


def mean_reciprocal_rank(ranks: Sequence[Optional[int]]) -> float:
    """Mean of ``1/rank`` of the first relevant result, 0 when there is none.

    Ranks are 1-based.
    """
    if not ranks:
        return 0.0
    total = 0.0
    for rank in ranks:
        if rank and rank > 0:
            total += 1.0 / float(rank)
    return total / float(len(ranks))


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: the ``ceil(pct/100 * n)``-th smallest value.

    No interpolation. With the 20-50 latency samples a goldset produces,
    interpolated percentiles invent precision that is not in the data.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    idx = math.ceil((pct / 100.0) * len(ordered)) - 1
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


# --------------------------------------------------------------------------
# Running an evaluation
# --------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Everything measured for one question."""

    qid: str
    question: str
    answerable: bool
    first_relevant_rank: Optional[int]
    retrieved_sources: List[str]
    refused: bool
    refusal_reason: str
    confidence: float
    groundedness: float
    citation_precision: Optional[float]
    content_hit: Optional[bool]
    latency_ms: float
    answer_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "answerable": self.answerable,
            "first_relevant_rank": self.first_relevant_rank,
            "retrieved_sources": self.retrieved_sources,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "confidence": round(self.confidence, 4),
            "groundedness": round(self.groundedness, 4),
            "citation_precision": (
                None if self.citation_precision is None else round(self.citation_precision, 4)
            ),
            "content_hit": self.content_hit,
            "latency_ms": round(self.latency_ms, 2),
            "answer": self.answer_text,
        }


@dataclass
class EvalReport:
    """Aggregate metrics plus every per-question record."""

    label: str
    n_questions: int
    n_answerable: int
    n_unanswerable: int
    recall_at: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    groundedness: float = 0.0
    citation_precision: float = 0.0
    content_accuracy: float = 0.0
    correct_refusals: float = 0.0
    false_refusals: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    index_chunks: int = 0
    results: List[QuestionResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_results: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "label": self.label,
            "questions": self.n_questions,
            "answerable": self.n_answerable,
            "unanswerable": self.n_unanswerable,
            "index_chunks": self.index_chunks,
            "recall_at": {str(k): round(v, 4) for k, v in sorted(self.recall_at.items())},
            "mrr": round(self.mrr, 4),
            "groundedness": round(self.groundedness, 4),
            "citation_precision": round(self.citation_precision, 4),
            "content_accuracy": round(self.content_accuracy, 4),
            "correct_refusals": round(self.correct_refusals, 4),
            "false_refusals": round(self.false_refusals, 4),
            "latency_ms": {
                "p50": round(self.latency_p50, 2),
                "p90": round(self.latency_p90, 2),
                "p95": round(self.latency_p95, 2),
            },
            "config": self.config,
        }
        if include_results:
            out["results"] = [r.to_dict() for r in self.results]
        return out

    def render(self) -> str:
        """Human-readable single-configuration summary."""
        lines: List[str] = []
        lines.append("=" * 68)
        lines.append("EVALUATION: %s" % self.label)
        lines.append("=" * 68)
        lines.append(
            "corpus index : %d chunks | %d questions (%d answerable, %d unanswerable)"
            % (self.index_chunks, self.n_questions, self.n_answerable, self.n_unanswerable)
        )
        lines.append("")
        lines.append("retrieval")
        for k in sorted(self.recall_at):
            lines.append("  recall@%-2d           %.3f" % (k, self.recall_at[k]))
        lines.append("  MRR                 %.3f" % self.mrr)
        lines.append("")
        lines.append("answers")
        lines.append("  groundedness        %.3f" % self.groundedness)
        lines.append("  citation precision  %.3f" % self.citation_precision)
        lines.append("  content accuracy    %.3f" % self.content_accuracy)
        lines.append("")
        lines.append("refusal behaviour")
        lines.append("  correct refusals    %.3f  (of %d unanswerable)" % (self.correct_refusals, self.n_unanswerable))
        lines.append("  false refusals      %.3f  (of %d answerable)" % (self.false_refusals, self.n_answerable))
        lines.append("")
        lines.append("latency")
        lines.append(
            "  p50 %.1f ms | p90 %.1f ms | p95 %.1f ms"
            % (self.latency_p50, self.latency_p90, self.latency_p95)
        )
        return "\n".join(lines)

    def failures(self) -> List[QuestionResult]:
        """Questions worth looking at by hand."""
        bad: List[QuestionResult] = []
        for r in self.results:
            if r.answerable and (r.refused or r.first_relevant_rank is None):
                bad.append(r)
            elif not r.answerable and not r.refused:
                bad.append(r)
        return bad


def run_eval(
    pipeline: FAQPipeline,
    goldset: GoldSet,
    *,
    k_values: Sequence[int] = (1, 3, 5),
    label: Optional[str] = None,
) -> EvalReport:
    """Run a goldset against an already-indexed pipeline.

    Retrieval metrics are measured on the retrieval stage directly, not on
    whatever survived the guardrails. Mixing the two hides the case where
    retrieval is fine and the refusal threshold is simply set too high.
    """
    k_values = tuple(sorted(set(int(k) for k in k_values)))
    max_k = max(k_values) if k_values else 5

    hits_at: Dict[int, List[bool]] = {k: [] for k in k_values}
    first_ranks: List[Optional[int]] = []
    grounded: List[float] = []
    cite_precisions: List[float] = []
    content_hits: List[bool] = []
    latencies: List[float] = []
    correct_refusals = 0
    false_refusals = 0
    results: List[QuestionResult] = []

    for gold in goldset.questions:
        retrieved = pipeline.retrieve(gold.question, max_k)
        first_rank: Optional[int] = None
        if gold.relevant_sources:
            for i, sc in enumerate(retrieved, start=1):
                if _is_relevant(sc, gold.relevant_sources):
                    first_rank = i
                    break

        started = time.perf_counter()
        answer: Answer = pipeline.ask(gold.question)
        elapsed = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed)

        if gold.answerable and gold.relevant_sources:
            for k in k_values:
                hits_at[k].append(first_rank is not None and first_rank <= k)
            first_ranks.append(first_rank)

        cite_precision: Optional[float] = None
        content_hit: Optional[bool] = None
        if gold.answerable:
            if answer.refused:
                false_refusals += 1
            else:
                grounded.append(answer.groundedness)
                if gold.relevant_sources and answer.citations:
                    good = sum(
                        1
                        for c in answer.citations
                        if any(marker in c.source for marker in gold.relevant_sources)
                    )
                    cite_precision = good / float(len(answer.citations))
                    cite_precisions.append(cite_precision)
                if gold.expected_contains:
                    low = answer.text.casefold()
                    content_hit = all(s.casefold() in low for s in gold.expected_contains)
                    content_hits.append(content_hit)
        else:
            if answer.refused:
                correct_refusals += 1

        results.append(
            QuestionResult(
                qid=gold.qid,
                question=gold.question,
                answerable=gold.answerable,
                first_relevant_rank=first_rank,
                retrieved_sources=[sc.chunk.source for sc in retrieved],
                refused=answer.refused,
                refusal_reason=answer.refusal_reason,
                confidence=answer.confidence,
                groundedness=answer.groundedness,
                citation_precision=cite_precision,
                content_hit=content_hit,
                latency_ms=elapsed,
                answer_text=answer.text,
            )
        )

    n_answerable = len(goldset.answerable)
    n_unanswerable = len(goldset.unanswerable)
    return EvalReport(
        label=label or pipeline.config.label,
        n_questions=len(goldset),
        n_answerable=n_answerable,
        n_unanswerable=n_unanswerable,
        recall_at={k: recall_at_k(hits_at[k]) for k in k_values},
        mrr=mean_reciprocal_rank(first_ranks),
        groundedness=statistics.fmean(grounded) if grounded else 0.0,
        citation_precision=statistics.fmean(cite_precisions) if cite_precisions else 0.0,
        content_accuracy=recall_at_k(content_hits),
        correct_refusals=(correct_refusals / n_unanswerable) if n_unanswerable else 0.0,
        false_refusals=(false_refusals / n_answerable) if n_answerable else 0.0,
        latency_p50=percentile(latencies, 50),
        latency_p90=percentile(latencies, 90),
        latency_p95=percentile(latencies, 95),
        index_chunks=len(pipeline.store),
        results=results,
        config=pipeline.config.describe(),
    )


def compare(
    configs: Sequence[PipelineConfig],
    goldset: GoldSet,
    *,
    corpus_paths: Optional[Sequence[str]] = None,
    k_values: Sequence[int] = (1, 3, 5),
) -> List[EvalReport]:
    """Index and evaluate the same goldset under several configurations.

    Each configuration gets a fresh pipeline and a fresh index, because a
    chunking change invalidates the index and reusing it would silently
    compare a new config against old chunks.
    """
    paths = list(corpus_paths or ([goldset.corpus] if goldset.corpus else []))
    if not paths:
        raise ValueError("no corpus path: pass corpus_paths or set 'corpus' in the goldset")
    reports: List[EvalReport] = []
    for cfg in configs:
        pipe = FAQPipeline(cfg)
        pipe.ingest(paths)
        reports.append(run_eval(pipe, goldset, k_values=k_values, label=cfg.label))
    return reports


def format_comparison_table(reports: Sequence[EvalReport]) -> str:
    """Fixed-width comparison table across configurations."""
    if not reports:
        return "(no reports)"
    k_values = sorted({k for r in reports for k in r.recall_at})
    headers = ["config", "chunks"]
    headers += ["r@%d" % k for k in k_values]
    headers += ["MRR", "grnd", "cite", "acc", "refuse", "false", "p50ms"]

    rows: List[List[str]] = []
    for r in reports:
        row = [r.label, str(r.index_chunks)]
        row += ["%.3f" % r.recall_at.get(k, 0.0) for k in k_values]
        row += [
            "%.3f" % r.mrr,
            "%.3f" % r.groundedness,
            "%.3f" % r.citation_precision,
            "%.3f" % r.content_accuracy,
            "%.3f" % r.correct_refusals,
            "%.3f" % r.false_refusals,
            "%.1f" % r.latency_p50,
        ]
        rows.append(row)

    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt(row) for row in rows]
    lines.append("")
    lines.append(
        "r@k = recall@k | grnd = groundedness | cite = citation precision | "
        "acc = expected-content accuracy"
    )
    lines.append(
        "refuse = correct refusals on unanswerable | false = wrong refusals on answerable"
    )
    return "\n".join(lines)
