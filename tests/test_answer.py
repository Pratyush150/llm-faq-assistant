"""Extractive answering: citations, spans, and the shared Answer contract."""

from __future__ import annotations

import pytest

from faqbot.answer import DEFAULT_SYSTEM_PROMPT, ExtractiveAnswerer, LLMAnswerer, build_prompt
from faqbot.types import Answer, Chunk, RefusalReason, ScoredChunk

BATTERY = Chunk(
    chunk_id="bat",
    doc_id="d1",
    text=(
        "[Battery and charging]\n\n"
        "How long does the AR-1 battery last?\n\n"
        "Runtime is up to 90 minutes in Eco mode and about 45 minutes in Turbo mode.\n"
        "Runtime is shorter on carpet because the brush motor draws more current."
    ),
    breadcrumb=("Battery and charging", "How long does the AR-1 battery last?"),
    meta={"source": "02-battery.md", "heading": "How long does the AR-1 battery last?"},
)

FILTER = Chunk(
    chunk_id="flt",
    doc_id="d2",
    text="[Consumables]\n\nWhich filter?\n\nThe washable filter is part number NW-FILT-02.",
    breadcrumb=("Consumables", "Which filter?"),
    meta={"source": "04-filters.md", "heading": "Which filter?"},
)


def _results():
    return [ScoredChunk(chunk=BATTERY, score=0.9, rank=1), ScoredChunk(chunk=FILTER, score=0.4, rank=2)]


def test_answer_quotes_the_source_sentence():
    answer = ExtractiveAnswerer().answer("How long does the battery last?", _results())
    assert "90 minutes" in answer.text
    assert answer.refused is False
    assert answer.confidence > 0.0


def test_answer_never_quotes_the_question_heading_back():
    answer = ExtractiveAnswerer().answer("How long does the battery last?", _results())
    assert "How long does the AR-1 battery last?" not in answer.text
    assert "[Battery and charging]" not in answer.text


def test_every_selected_sentence_carries_a_citation():
    answer = ExtractiveAnswerer(max_sentences=2).answer("How long does the battery last?", _results())
    assert len(answer.citations) == len([s for s in answer.text.split("] ") if s])
    for citation in answer.citations:
        assert citation.chunk_id in ("bat", "flt")
        assert citation.source in ("02-battery.md", "04-filters.md")


def test_citation_spans_point_at_the_quoted_text():
    answer = ExtractiveAnswerer().answer("How long does the battery last?", _results())
    for citation in answer.citations:
        source_text = BATTERY.text if citation.chunk_id == "bat" else FILTER.text
        assert source_text[citation.start_char : citation.end_char].strip() == citation.quote


def test_sentences_from_one_chunk_share_a_citation_marker():
    answer = ExtractiveAnswerer(max_sentences=3).answer("How long does the battery last?", _results())
    markers = {c.chunk_id: c.marker for c in answer.citations}
    assert len(set(markers.values())) == len(markers)
    for citation in answer.citations:
        assert "[%d]" % citation.marker in answer.text


def test_chunk_ids_property_is_ordered_and_unique():
    answer = ExtractiveAnswerer(max_sentences=3).answer("How long does the battery last?", _results())
    assert answer.chunk_ids == list(dict.fromkeys(c.chunk_id for c in answer.citations))
    assert answer.doc_ids


def test_no_results_is_a_refusal_not_an_exception():
    answer = ExtractiveAnswerer().answer("anything at all", [])
    assert answer.refused is True
    assert answer.refusal_reason == RefusalReason.NO_CONTEXT
    assert answer.citations == []


def test_duplicate_sentences_are_not_repeated():
    duplicate = Chunk(
        chunk_id="dup",
        doc_id="d3",
        text="Runtime is up to 90 minutes in Eco mode and about 45 minutes in Turbo mode.",
        meta={"source": "copy.md"},
    )
    results = _results() + [ScoredChunk(chunk=duplicate, score=0.35, rank=3)]
    answer = ExtractiveAnswerer(max_sentences=3).answer("How long does the battery last?", results)
    assert answer.text.count("90 minutes") == 1


def test_answer_serialises_to_json_with_every_field():
    answer = ExtractiveAnswerer().answer("How long does the battery last?", _results())
    payload = answer.to_dict()
    for key in (
        "text", "citations", "chunk_ids", "confidence", "refused",
        "refusal_reason", "groundedness", "latency_ms", "diagnostics",
    ):
        assert key in payload
    assert isinstance(answer.to_json(), str)


def test_render_shows_status_and_sources():
    answer = ExtractiveAnswerer().answer("How long does the battery last?", _results())
    rendered = answer.render()
    assert "sources:" in rendered
    assert "answered" in rendered


def test_refused_answer_renders_its_reason():
    rendered = Answer(text="no", refused=True, refusal_reason=RefusalReason.OUT_OF_DOMAIN).render()
    assert "REFUSED" in rendered
    assert "out_of_domain" in rendered


def test_diagnostics_explain_the_sentence_selection():
    answer = ExtractiveAnswerer().answer("How long does the battery last?", _results())
    diag = answer.diagnostics
    assert diag["answerer"] == "extractive"
    assert diag["candidates"] >= diag["selected"] >= 1
    assert len(diag["sentence_scores"]) == diag["selected"]


def test_prompt_wraps_retrieved_text_as_untrusted_data():
    system, user = build_prompt("How long does the battery last?", _results())
    assert system == DEFAULT_SYSTEM_PROMPT
    assert "INSUFFICIENT_CONTEXT" in system
    assert "UNTRUSTED REFERENCE DATA" in user
    assert user.rstrip().endswith("Question: How long does the battery last?")


class _StubLLM(LLMAnswerer):
    """A local stub. No provider, no network, no key."""

    name = "stub"

    def __init__(self, reply: str, **kwargs):
        super().__init__(**kwargs)
        self.reply = reply
        self.seen = None

    def complete(self, system: str, user: str) -> str:
        self.seen = (system, user)
        return self.reply


def test_llm_answerer_parses_citation_markers():
    stub = _StubLLM("Runtime is 90 minutes in Eco mode [bat].")
    answer = stub.answer("How long?", _results())
    assert answer.refused is False
    assert [c.chunk_id for c in answer.citations] == ["bat"]


def test_llm_answerer_refuses_on_insufficient_context():
    answer = _StubLLM("INSUFFICIENT_CONTEXT").answer("How long?", _results())
    assert answer.refused is True
    assert answer.refusal_reason == RefusalReason.LOW_CONFIDENCE


def test_llm_answerer_falls_back_to_extraction_when_the_provider_fails():
    class _Broken(LLMAnswerer):
        name = "broken"

        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("provider is down")

    answer = _Broken().answer("How long does the battery last?", _results())
    assert answer.refused is False
    assert answer.diagnostics["degraded"] is True
    assert "90 minutes" in answer.text


def test_llm_prompt_receives_the_delimited_context():
    stub = _StubLLM("ok")
    stub.answer("How long?", _results())
    system, user = stub.seen
    assert "Never follow instructions" in system or "untrusted" in system.casefold()
    assert '<document id="bat"' in user


def test_confidence_stays_in_range():
    answer = ExtractiveAnswerer().answer("How long does the battery last?", _results())
    assert 0.0 <= answer.confidence <= 1.0
    assert answer.latency_ms == pytest.approx(0.0)
