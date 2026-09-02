"""Pipeline: end to end, conversation memory, query rewriting, persistence."""

from __future__ import annotations

import os

import pytest

from faqbot.pipeline import (
    ConversationMemory,
    FAQPipeline,
    PipelineConfig,
    QueryRewriter,
    Turn,
    extract_entities,
)
from faqbot.types import RefusalReason


# -- query rewriting ---------------------------------------------------------


def _memory_after(question: str, answer: str = "Up to 90 minutes in Eco mode.") -> ConversationMemory:
    memory = ConversationMemory()
    memory.add(
        Turn(
            question=question,
            rewritten=question,
            answer=answer,
            entities=tuple(extract_entities(question)),
        )
    )
    return memory


def test_entities_include_model_and_part_numbers():
    found = extract_entities("Does the AR-1 Pro use the NW-FILT-02 filter?")
    assert "AR-1 Pro" in found or "AR-1" in found
    assert "NW-FILT-02" in found


def test_pronoun_followup_is_resolved():
    """'it' has no meaning to a retriever; the previous turn supplies the subject."""
    memory = _memory_after("How long does the AR-1 battery last?")
    rewritten, changed, info = QueryRewriter().rewrite("Does it work on carpet?", memory)
    assert changed is True
    assert info["rule"] == "pronoun"
    assert rewritten == "Does the AR-1 work on carpet?"
    assert "it" not in rewritten.split()


def test_topic_switch_reuses_the_previous_question():
    memory = _memory_after("How long does the AR-1 battery last?")
    rewritten, changed, info = QueryRewriter().rewrite("What about the Pro version?", memory)
    assert changed is True
    assert info["rule"] == "topic_switch"
    assert "Pro version" in rewritten
    assert "How long" in rewritten


def test_standalone_questions_are_left_alone():
    memory = _memory_after("How long does the AR-1 battery last?")
    question = "How do I replace the filter NW-FILT-02?"
    rewritten, changed, _ = QueryRewriter().rewrite(question, memory)
    assert changed is False
    assert rewritten == question


def test_nothing_is_rewritten_without_history():
    rewritten, changed, _ = QueryRewriter().rewrite("Does it work on carpet?", ConversationMemory())
    assert changed is False
    assert rewritten == "Does it work on carpet?"


def test_rewriting_can_be_disabled():
    memory = _memory_after("How long does the AR-1 battery last?")
    _, changed, _ = QueryRewriter(enabled=False).rewrite("Does it work on carpet?", memory)
    assert changed is False


def test_memory_is_bounded():
    memory = ConversationMemory(max_turns=2)
    for i in range(5):
        memory.add(Turn(question="q%d" % i, rewritten="q%d" % i, answer="a"))
    assert len(memory) == 2
    assert memory.last().question == "q4"


# -- end to end --------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline(request):
    corpus = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus"
    )
    if not os.path.isdir(corpus):
        pytest.skip("bundled corpus is not present")
    pipe = FAQPipeline(PipelineConfig(label="test"))
    pipe.ingest([corpus])
    return pipe


def test_index_is_built(pipeline):
    stats = pipeline.stats()
    assert stats["chunks"] > 40
    assert stats["documents"] > 10
    assert stats["bm25_chunks"] == stats["chunks"]
    assert stats["vocabulary_terms"] > 300


def test_answers_a_factual_question_with_a_citation(pipeline):
    answer = pipeline.ask("How long does the AR-1 battery last on one charge?")
    assert answer.refused is False
    assert "90 minutes" in answer.text
    assert answer.citations
    assert any("02-battery" in c.source for c in answer.citations)
    assert answer.groundedness == pytest.approx(1.0)


def test_answers_an_identifier_question(pipeline):
    answer = pipeline.ask("Which filter part number does the AR-1 use?")
    assert answer.refused is False
    assert "NW-FILT-02" in answer.text


def test_refuses_out_of_domain(pipeline):
    answer = pipeline.ask("What is the best recipe for sourdough bread?")
    assert answer.refused is True
    assert answer.refusal_reason == RefusalReason.OUT_OF_DOMAIN
    assert answer.citations == []


def test_conversation_followup_end_to_end(pipeline):
    pipeline.reset_session("t1")
    first = pipeline.ask("How long does the AR-1 battery last?", session_id="t1")
    assert first.refused is False
    second = pipeline.ask("Does it work on carpet?", session_id="t1")
    assert second.rewritten_question == "Does the AR-1 work on carpet?"
    assert second.refused is False
    assert "carpet" in second.text.casefold()


def test_latency_is_recorded(pipeline):
    answer = pipeline.ask("What is the dust bin capacity?")
    assert answer.latency_ms > 0.0


def test_reingesting_the_same_corpus_does_not_grow_the_index(pipeline, corpus_dir):
    before = len(pipeline.store)
    stats = pipeline.ingest([corpus_dir])
    assert len(pipeline.store) == before
    assert stats["chunks_added"] == 0
    assert stats["chunks_replaced"] == before


def test_deleting_a_document_removes_its_chunks(corpus_dir):
    pipe = FAQPipeline(PipelineConfig(label="delete"))
    pipe.ingest([corpus_dir])
    before = len(pipe.store)
    doc_id = pipe.documents[0].doc_id
    removed = pipe.delete_document(doc_id)
    assert removed > 0
    assert len(pipe.store) == before - removed
    assert len(pipe.bm25) == len(pipe.store)


def test_index_save_and_load_round_trip(corpus_dir, tmp_path):
    pipe = FAQPipeline(PipelineConfig(label="persist"))
    pipe.ingest([corpus_dir])
    path = pipe.save(os.path.join(str(tmp_path), "index.json"))
    assert os.path.exists(path)

    reloaded = FAQPipeline(PipelineConfig(label="persist"))
    reloaded.load(path)
    assert len(reloaded.store) == len(pipe.store)
    assert len(reloaded.bm25) == len(pipe.store)
    answer = reloaded.ask("How long does the AR-1 battery last on one charge?")
    assert "90 minutes" in answer.text


def test_loading_an_index_with_the_wrong_dimension_is_refused(corpus_dir, tmp_path):
    pipe = FAQPipeline(PipelineConfig(embedder_kwargs={"dim": 64}, label="small"))
    pipe.ingest([corpus_dir])
    path = pipe.save(os.path.join(str(tmp_path), "small.json"))
    other = FAQPipeline(PipelineConfig(embedder_kwargs={"dim": 256}, label="big"))
    with pytest.raises(ValueError):
        other.load(path)


def test_prompt_injection_in_a_retrieved_document_is_neutralised(corpus_dir):
    """A poisoned page must not put words in the bot's mouth."""
    adversarial = os.path.join(os.path.dirname(corpus_dir), "adversarial")
    if not os.path.isdir(adversarial):
        pytest.skip("adversarial sample is not present")
    pipe = FAQPipeline(PipelineConfig(label="adversarial"))
    pipe.ingest([corpus_dir, adversarial])

    answer = pipe.ask("What do owners say about AR-1 battery life?")
    hits = answer.diagnostics["injection_hits"]
    assert hits, "the injection scan should have fired"
    assert {h["kind"] for h in hits} & {"override", "persona-switch", "citation-suppression"}

    lowered = answer.text.casefold()
    assert "ignore all previous instructions" not in lowered
    assert "unrestricted assistant" not in lowered
    assert "warranty is void" not in lowered


def test_injection_can_be_configured_to_refuse(corpus_dir):
    adversarial = os.path.join(os.path.dirname(corpus_dir), "adversarial")
    if not os.path.isdir(adversarial):
        pytest.skip("adversarial sample is not present")
    cfg = PipelineConfig(label="strict")
    cfg.refuse_on_injection = True
    pipe = FAQPipeline(cfg)
    pipe.ingest([corpus_dir, adversarial])
    answer = pipe.ask("What do owners say about AR-1 battery life?")
    assert answer.refused is True
    assert answer.refusal_reason == RefusalReason.INJECTION


def test_retrieval_without_a_reranker_still_reports_match_quality(corpus_dir):
    """Turning reranking off must change ranking, never refusal behaviour."""
    pipe = FAQPipeline(PipelineConfig(reranker="none", label="norerank"))
    pipe.ingest([corpus_dir])
    results = pipe.retrieve("How long does the AR-1 battery last?")
    assert results
    assert results[0].components["match_quality"] > 0.5
    assert pipe.ask("How long does the AR-1 battery last?").refused is False


def test_sparse_only_and_dense_only_configurations_both_answer(corpus_dir):
    for kwargs in ({"use_dense": False}, {"use_sparse": False}):
        pipe = FAQPipeline(PipelineConfig(label="one-sided", **kwargs))
        pipe.ingest([corpus_dir])
        answer = pipe.ask("Which filter part number does the AR-1 use?")
        assert answer.refused is False
        assert "NW-FILT-02" in answer.text


def test_a_pipeline_with_no_retriever_enabled_is_rejected(corpus_dir):
    pipe = FAQPipeline(PipelineConfig(use_dense=False, use_sparse=False, label="broken"))
    pipe.ingest([corpus_dir])
    with pytest.raises(ValueError):
        pipe.retrieve("anything")
