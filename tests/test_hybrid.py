"""Hybrid retrieval: RRF fusion and what it buys over either retriever alone."""

from __future__ import annotations

import pytest

from faqbot.embedding import HashingEmbedder
from faqbot.store import BM25Index, HybridRetriever, VectorStore, reciprocal_rank_fusion
from faqbot.types import Chunk, ScoredChunk


def _sc(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=chunk_id, doc_id="d", text=chunk_id), score=score)


def test_rrf_ranks_a_doc_found_by_both_retrievers_above_either_alone():
    """The property that makes hybrid retrieval worth the complexity."""
    dense = [_sc("both", 0.9), _sc("dense_only", 0.88)]
    sparse = [_sc("sparse_only", 9.1), _sc("both", 8.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    order = [f.chunk.chunk_id for f in fused]
    assert order[0] == "both"
    assert set(order[1:]) == {"dense_only", "sparse_only"}


def test_rrf_scores_match_the_formula():
    dense = [_sc("a", 1.0), _sc("b", 0.5)]
    sparse = [_sc("b", 3.0)]
    fused = {f.chunk.chunk_id: f.score for f in reciprocal_rank_fusion([dense, sparse], k=60)}
    assert fused["a"] == pytest.approx(1.0 / 61.0)
    assert fused["b"] == pytest.approx(1.0 / 62.0 + 1.0 / 61.0)
    assert fused["b"] > fused["a"]


def test_rrf_records_which_retrievers_found_each_chunk():
    fused = reciprocal_rank_fusion([[_sc("a", 1.0)], [_sc("a", 2.0), _sc("b", 1.0)]], k=60)
    by_id = {f.chunk.chunk_id: f for f in fused}
    assert by_id["a"].components["retrievers"] == 2.0
    assert by_id["b"].components["retrievers"] == 1.0


def test_rrf_weights_shift_the_balance():
    dense = [_sc("d", 1.0)]
    sparse = [_sc("s", 1.0)]
    dense_heavy = reciprocal_rank_fusion([dense, sparse], k=60, weights=[3.0, 1.0])
    sparse_heavy = reciprocal_rank_fusion([dense, sparse], k=60, weights=[1.0, 3.0])
    assert dense_heavy[0].chunk.chunk_id == "d"
    assert sparse_heavy[0].chunk.chunk_id == "s"


def test_rrf_rejects_a_weight_length_mismatch():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[_sc("a", 1.0)]], k=60, weights=[1.0, 1.0])


def test_rrf_ignores_score_scale_entirely():
    """Ranks, not scores: a retriever cannot win by inflating its numbers."""
    small = [_sc("x", 0.001), _sc("y", 0.0005)]
    huge = [_sc("y", 5000.0), _sc("x", 4000.0)]
    a = reciprocal_rank_fusion([small, huge], k=60)
    b = reciprocal_rank_fusion([[_sc("x", 1.0), _sc("y", 0.9)], [_sc("y", 1.0), _sc("x", 0.9)]], k=60)
    assert [s.score for s in a] == [pytest.approx(s.score) for s in b]


CHUNKS = [
    Chunk(chunk_id="parts", doc_id="d1", breadcrumb=("Consumables",),
          text="The replacement filter is part number NW-FILT-02 and it is washable."),
    Chunk(chunk_id="runtime", doc_id="d2", breadcrumb=("Battery",),
          text="The robot runs for up to 90 minutes in Eco mode before it returns to the dock."),
    Chunk(chunk_id="filters", doc_id="d3", breadcrumb=("Maintenance",),
          text="Rinse the pleated filter under cold water and let it dry fully before refitting it."),
]


def _hybrid():
    embedder = HashingEmbedder(dim=256)
    store = VectorStore(embedder.dim, embedder_name=embedder.name)
    store.add(CHUNKS, embedder.embed([c.text for c in CHUNKS]))
    bm25 = BM25Index()
    bm25.add(CHUNKS)
    return HybridRetriever(store=store, bm25=bm25, embedder=embedder)


def test_hybrid_retrieval_finds_an_exact_part_number():
    hits = _hybrid().retrieve("NW-FILT-02", k=3)
    assert hits[0].chunk.chunk_id == "parts"
    assert hits[0].components["bm25"] > 0.0


def test_hybrid_results_carry_both_retriever_scores():
    hits = _hybrid().retrieve("how long does it run in eco mode", k=3)
    assert hits[0].chunk.chunk_id == "runtime"
    for hit in hits:
        assert "dense" in hit.components
        assert "bm25" in hit.components
        assert "rrf" in hit.components


def test_hybrid_handles_an_empty_query():
    assert _hybrid().retrieve("   ", k=3) == []


def test_hybrid_describe_reports_index_sizes():
    info = _hybrid().describe()
    assert info["dense_chunks"] == 3
    assert info["sparse_chunks"] == 3
    assert info["embedder"] == "hashing"
