"""Reranking: explainable features, and the rare-term signal that matters most."""

from __future__ import annotations

import pytest

from faqbot.rerank import (
    CrossEncoderReranker,
    FeatureReranker,
    IdentityReranker,
    RerankWeights,
    compute_match_quality,
    get_reranker,
)
from faqbot.store import BM25Index
from faqbot.types import Chunk, ScoredChunk

E03 = Chunk(
    chunk_id="e03", doc_id="d1", index=3, breadcrumb=("Error codes", "E03 — airflow blocked"),
    text="[Error codes]\n\nE03 — airflow blocked\n\nThe filter is wet, clogged or missing, or the bin is full.",
)
E09 = Chunk(
    chunk_id="e09", doc_id="d1", index=9, breadcrumb=("Error codes", "E09 — bin not detected"),
    text="[Error codes]\n\nE09 — bin not detected\n\nThe dust bin is out, or its latch magnet is dirty.",
)
SUPPORT = Chunk(
    chunk_id="sup", doc_id="d2", index=0, breadcrumb=("Support", "What should I try first?"),
    text="[Support]\n\nWhat should I try first?\n\nThose four steps clear most E01, E02, E03 and E04 reports.",
)


def _index() -> BM25Index:
    index = BM25Index()
    index.add([E03, E09, SUPPORT])
    return index


def test_reranker_pulls_the_exact_error_code_to_the_top():
    """The retriever put E09 first; the rare-term feature has to undo that."""
    candidates = [
        ScoredChunk(chunk=E09, score=0.031),
        ScoredChunk(chunk=SUPPORT, score=0.029),
        ScoredChunk(chunk=E03, score=0.016),
    ]
    ranked = FeatureReranker(idf=_index().idf).rerank("What does error code E03 mean?", candidates, k=3)
    assert ranked[0].chunk.chunk_id == "e03"
    assert ranked[0].rank == 1


def test_rare_term_feature_favours_the_exact_identifier():
    reranker = FeatureReranker(idf=_index().idf)
    query = "What does error code E03 mean?"
    good = reranker.features(query, ScoredChunk(chunk=E03, score=0.0))
    bad = reranker.features(query, ScoredChunk(chunk=E09, score=0.0))
    assert good["rare_term"] > bad["rare_term"]
    assert good["lexical"] >= bad["lexical"]


def test_every_result_explains_its_own_score():
    ranked = FeatureReranker(idf=_index().idf).rerank("error code E03", [ScoredChunk(chunk=E03, score=0.1)], k=1)
    components = ranked[0].components
    for key in ("f_lexical", "f_rare_term", "f_breadcrumb", "f_exact_phrase", "w_rare_term"):
        assert key in components
    assert components["rerank"] == pytest.approx(ranked[0].score)
    assert components["pre_rerank_score"] == pytest.approx(0.1)


def test_match_quality_is_absolute_and_bounded():
    good = compute_match_quality("What does error code E03 mean?", E03, idf=_index().idf)
    bad = compute_match_quality("What is the warranty period?", E03, idf=_index().idf)
    assert 0.0 <= bad < good <= 1.0


def test_match_quality_works_without_an_idf_source():
    value = compute_match_quality("error code E03", E03)
    assert 0.0 < value <= 1.0


def test_breadcrumb_match_is_rewarded():
    reranker = FeatureReranker()
    feats = reranker.features("airflow blocked", ScoredChunk(chunk=E03, score=0.0))
    assert feats["breadcrumb"] > 0.0


def test_exact_phrase_bonus_beats_scattered_words():
    reranker = FeatureReranker()
    phrase = reranker.features("the bin is full", ScoredChunk(chunk=E03, score=0.0))
    scattered = reranker.features("bin full wet", ScoredChunk(chunk=E03, score=0.0))
    assert phrase["exact_phrase"] > scattered["exact_phrase"]


def test_long_chunks_take_a_length_penalty():
    long_chunk = Chunk(chunk_id="long", doc_id="d", text=" ".join(["word"] * 500))
    feats = FeatureReranker(target_tokens=100).features("word", ScoredChunk(chunk=long_chunk, score=0.0))
    assert feats["length_penalty"] == pytest.approx(1.0)


def test_question_shape_defaults_to_zero_weight():
    """Switched off because the eval harness said it hurt, not by oversight."""
    assert RerankWeights().question_shape == 0.0


def test_identity_reranker_preserves_order():
    candidates = [ScoredChunk(chunk=E09, score=0.9), ScoredChunk(chunk=E03, score=0.1)]
    ranked = IdentityReranker().rerank("anything", candidates, k=2)
    assert [r.chunk.chunk_id for r in ranked] == ["e09", "e03"]
    assert [r.rank for r in ranked] == [1, 2]


def test_empty_candidate_list_is_handled():
    assert FeatureReranker().rerank("q", [], k=5) == []


def test_cross_encoder_degrades_to_the_feature_reranker_when_absent():
    reranker = CrossEncoderReranker(fallback=FeatureReranker(idf=_index().idf))
    ranked = reranker.rerank(
        "What does error code E03 mean?",
        [ScoredChunk(chunk=E09, score=0.03), ScoredChunk(chunk=E03, score=0.01)],
        k=2,
    )
    assert len(ranked) == 2
    assert isinstance(CrossEncoderReranker.is_available(), bool)


def test_registry_lookup():
    assert isinstance(get_reranker("feature"), FeatureReranker)
    assert isinstance(get_reranker("identity"), IdentityReranker)
    with pytest.raises(KeyError):
        get_reranker("nope")
