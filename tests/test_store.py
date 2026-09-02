"""Vector store and BM25, including hand-computed BM25 values.

The BM25 test is deliberately arithmetic rather than "the right document comes
first". A scoring function that ranks correctly but is not BM25 will silently
misbehave the first time the corpus changes shape, and no ranking assertion
would catch it.
"""

from __future__ import annotations

import math
import os

import pytest

from faqbot.embedding import HashingEmbedder
from faqbot.store import BM25Index, VectorStore
from faqbot.types import Chunk


def _hand_bm25(idf: float, tf: int, dl: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    return idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))


def test_bm25_statistics_match_the_toy_corpus(toy_chunks):
    index = BM25Index()
    index.add(toy_chunks)
    assert len(index) == 3
    # Token counts after stopword removal: 4, 4, 2.
    assert index.avgdl == pytest.approx(10.0 / 3.0)


def test_bm25_idf_matches_the_formula(toy_chunks):
    index = BM25Index()
    index.add(toy_chunks)
    n = 3
    # "fox" appears in c1 and c3.
    assert index.idf("fox") == pytest.approx(math.log(1.0 + (n - 2 + 0.5) / (2 + 0.5)))
    # "quick" appears in c1 only.
    assert index.idf("quick") == pytest.approx(math.log(1.0 + (n - 1 + 0.5) / (1 + 0.5)))
    # A term that is not in the corpus gets the maximum idf.
    assert index.idf("aardvark") == pytest.approx(math.log(1.0 + (n + 0.5) / 0.5))


def test_bm25_score_matches_a_hand_computed_value(toy_chunks):
    index = BM25Index()
    index.add(toy_chunks)
    avgdl = 10.0 / 3.0
    idf_fox = math.log(1.0 + 1.5 / 2.5)

    expected_c1 = _hand_bm25(idf_fox, tf=1, dl=4, avgdl=avgdl)
    expected_c3 = _hand_bm25(idf_fox, tf=1, dl=2, avgdl=avgdl)

    assert index.score("fox", "c1") == pytest.approx(expected_c1)
    assert index.score("fox", "c3") == pytest.approx(expected_c3)
    # Length normalisation: the shorter chunk scores higher for the same tf.
    assert expected_c3 > expected_c1


def test_bm25_multi_term_score_is_the_sum_of_term_scores(toy_chunks):
    index = BM25Index()
    index.add(toy_chunks)
    combined = index.score("fox dog", "c3")
    assert combined == pytest.approx(index.score("fox", "c3") + index.score("dog", "c3"))


def test_bm25_search_orders_by_score_and_drops_zeros(toy_chunks):
    index = BM25Index()
    index.add(toy_chunks)
    results = index.search("fox", k=5)
    assert [r.chunk.chunk_id for r in results] == ["c3", "c1"]
    assert results[0].rank == 1
    assert results[0].components["bm25"] == pytest.approx(results[0].score)
    assert index.search("aardvark", k=5) == []


def test_bm25_delete_document_updates_document_frequency(toy_chunks):
    index = BM25Index()
    index.add(toy_chunks)
    assert index.delete_document("d1") == 1
    assert len(index) == 2
    # "fox" now appears in one of two chunks.
    assert index.idf("fox") == pytest.approx(math.log(1.0 + (2 - 1 + 0.5) / 1.5))
    assert index.idf("quick") == pytest.approx(math.log(1.0 + 2.5 / 0.5))


def test_bm25_keeps_identifier_tokens_whole():
    index = BM25Index()
    index.add([
        Chunk(chunk_id="a", doc_id="d", text="Order the NW-FILT-02 replacement filter."),
        Chunk(chunk_id="b", doc_id="d", text="Order the NW-BRUSH-11 replacement side brush."),
    ])
    hits = index.search("NW-FILT-02", k=2)
    assert [h.chunk.chunk_id for h in hits] == ["a"]


def _store_with(chunks):
    embedder = HashingEmbedder(dim=128)
    store = VectorStore(embedder.dim, embedder_name=embedder.name)
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store, embedder


def test_vector_store_cosine_search_ranks_the_right_chunk_first():
    chunks = [
        Chunk(chunk_id="bat", doc_id="d1", text="The AR-1 runs for 90 minutes in Eco mode."),
        Chunk(chunk_id="warr", doc_id="d2", text="The warranty is 24 months on the robot and dock."),
    ]
    store, embedder = _store_with(chunks)
    hits = store.search(embedder.embed_one("How long does it run in Eco mode?"), k=2)
    assert hits[0].chunk.chunk_id == "bat"
    assert hits[0].rank == 1
    assert 0.0 <= hits[0].score <= 1.0


def test_metadata_filtering_supports_equality_membership_and_predicates():
    chunks = [
        Chunk(chunk_id="a", doc_id="d1", text="Filter guidance.", meta={"kind": "faq_pair", "row": 1}),
        Chunk(chunk_id="b", doc_id="d2", text="Filter guidance.", meta={"kind": "html", "row": 9}),
    ]
    store, embedder = _store_with(chunks)
    query = embedder.embed_one("filter")
    assert [h.chunk.chunk_id for h in store.search(query, 5, where={"kind": "html"})] == ["b"]
    assert len(store.search(query, 5, where={"kind": ["faq_pair", "html"]})) == 2
    assert [h.chunk.chunk_id for h in store.search(query, 5, where={"row": lambda v: v > 5})] == ["b"]


def test_upsert_document_replaces_all_of_its_chunks():
    old = [
        Chunk(chunk_id="o1", doc_id="d1", text="Runtime is 60 minutes."),
        Chunk(chunk_id="o2", doc_id="d1", text="Charging takes 300 minutes."),
    ]
    store, embedder = _store_with(old)
    assert len(store) == 2
    new = [Chunk(chunk_id="n1", doc_id="d1", text="Runtime is 90 minutes.")]
    removed, added = store.upsert_document("d1", new, embedder.embed([c.text for c in new]))
    assert (removed, added) == (2, 1)
    assert len(store) == 1
    assert store.get("o1") is None
    assert store.get("n1") is not None


def test_upsert_rejects_chunks_from_another_document():
    store, embedder = _store_with([Chunk(chunk_id="a", doc_id="d1", text="Body text.")])
    stray = [Chunk(chunk_id="x", doc_id="OTHER", text="Body text.")]
    with pytest.raises(ValueError):
        store.upsert_document("d1", stray, embedder.embed(["Body text."]))


def test_delete_document_removes_chunks_and_is_idempotent():
    store, _ = _store_with([
        Chunk(chunk_id="a", doc_id="d1", text="One body."),
        Chunk(chunk_id="b", doc_id="d1", text="Two body."),
        Chunk(chunk_id="c", doc_id="d2", text="Three body."),
    ])
    assert store.delete_document("d1") == 2
    assert store.delete_document("d1") == 0
    assert len(store) == 1
    assert store.doc_ids == ["d2"]


def test_vector_dimension_mismatch_is_rejected():
    store = VectorStore(8)
    with pytest.raises(ValueError):
        store.add([Chunk(chunk_id="a", doc_id="d", text="x")], [[0.0] * 4])
    with pytest.raises(ValueError):
        store.search([0.0] * 4, 1)


def test_json_round_trip_preserves_chunks_and_vectors(tmp_path):
    chunks = [
        Chunk(chunk_id="a", doc_id="d1", text="Runtime is 90 minutes.", breadcrumb=("Battery", "Runtime")),
        Chunk(chunk_id="b", doc_id="d2", text="Warranty is 24 months.", meta={"kind": "md"}),
    ]
    store, _ = _store_with(chunks)
    path = store.save_json(os.path.join(str(tmp_path), "idx", "index.json"))
    loaded = VectorStore.load_json(path)
    assert len(loaded) == 2
    assert loaded.dim == store.dim
    assert loaded.embedder_name == store.embedder_name
    assert loaded.get("a").breadcrumb == ("Battery", "Runtime")
    assert loaded.vector("a") == store.vector("a")


def test_loading_an_unknown_index_format_is_refused():
    with pytest.raises(ValueError):
        VectorStore.from_dict({"format": 999, "dim": 4, "chunks": [], "vectors": []})


@pytest.mark.skipif(not VectorStore.npz_available(), reason="numpy is not installed")
def test_npz_round_trip_matches_json(tmp_path):
    chunks = [Chunk(chunk_id="a", doc_id="d1", text="Runtime is 90 minutes.")]
    store, _ = _store_with(chunks)
    path = store.save_npz(os.path.join(str(tmp_path), "index.npz"))
    loaded = VectorStore.load_npz(path)
    assert len(loaded) == 1
    assert loaded.get("a").text == "Runtime is 90 minutes."
    for got, want in zip(loaded.vector("a"), store.vector("a")):
        assert got == pytest.approx(want, abs=1e-6)


def test_stats_report_index_shape(toy_chunks):
    store, _ = _store_with(toy_chunks)
    stats = store.stats()
    assert stats["chunks"] == 3
    assert stats["documents"] == 3
    assert stats["dim"] == 128
    assert stats["mean_chunk_chars"] > 0
