"""Embedding: determinism, unit norm, and capability detection.

The hashing embedder is the default because it makes everything else testable.
These tests pin the two properties the rest of the system relies on.
"""

from __future__ import annotations

import math

from faqbot.embedding import (
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    available_embedders,
    cosine,
    embedder_capabilities,
    get_embedder,
    l2_norm,
    register_embedder,
)

TEXTS = [
    "How long does the AR-1 battery last?",
    "The filter part number is NW-FILT-02.",
    "Charging takes about 240 minutes from empty.",
]


def test_hashing_embedder_is_deterministic():
    a = HashingEmbedder(dim=128).embed(TEXTS)
    b = HashingEmbedder(dim=128).embed(TEXTS)
    assert a == b


def test_hashing_embedder_is_deterministic_across_instances_and_calls():
    e = HashingEmbedder(dim=64)
    first = e.embed_one(TEXTS[0])
    second = e.embed_one(TEXTS[0])
    assert first == second
    assert first == HashingEmbedder(dim=64).embed_one(TEXTS[0])


def test_vectors_are_l2_normalised():
    for vec in HashingEmbedder(dim=256).embed(TEXTS):
        assert abs(l2_norm(vec) - 1.0) < 1e-9


def test_empty_text_gives_a_zero_vector_not_a_crash():
    vec = HashingEmbedder(dim=32).embed_one("")
    assert len(vec) == 32
    assert l2_norm(vec) in (0.0, 1.0)


def test_dimension_is_respected():
    for dim in (16, 64, 512):
        assert len(HashingEmbedder(dim=dim).embed_one("hello")) == dim
        assert HashingEmbedder(dim=dim).dim == dim


def test_cosine_of_a_vector_with_itself_is_one():
    vec = HashingEmbedder(dim=128).embed_one(TEXTS[0])
    assert abs(cosine(vec, vec) - 1.0) < 1e-9


def test_similar_texts_score_higher_than_unrelated_ones():
    e = HashingEmbedder(dim=256)
    battery = e.embed_one("How long does the AR-1 battery last?")
    battery2 = e.embed_one("How long does the battery of the AR-1 last?")
    filters = e.embed_one("Rinse the pleated filter under cold water.")
    assert cosine(battery, battery2) > cosine(battery, filters)


def test_character_ngrams_survive_a_typo_in_a_part_number():
    """A one-character slip must not drop the match to zero."""
    e = HashingEmbedder(dim=256)
    exact = e.embed_one("NW-FILT-02 replacement filter")
    typo = e.embed_one("NW-FILT-O2 replacement filter")
    unrelated = e.embed_one("The warranty runs for 24 months.")
    assert cosine(exact, typo) > 0.5
    assert cosine(exact, typo) > cosine(exact, unrelated)


def test_seed_changes_the_projection():
    a = HashingEmbedder(dim=64, seed=0).embed_one(TEXTS[0])
    b = HashingEmbedder(dim=64, seed=7).embed_one(TEXTS[0])
    assert a != b


def test_cosine_rejects_mismatched_dimensions():
    try:
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")


def test_invalid_construction_is_rejected():
    for kwargs in ({"dim": 0}, {"ngram_range": (0, 3)}, {"ngram_range": (5, 2)}):
        try:
            HashingEmbedder(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError("expected ValueError for %r" % kwargs)  # pragma: no cover


def test_hashing_is_always_available_and_offline():
    caps = embedder_capabilities()
    assert caps["hashing"]["available"] is True
    assert caps["hashing"]["offline"] is True
    assert "hashing" in available_embedders()


def test_optional_backends_are_guarded_not_imported():
    """Constructing a hosted embedder must not touch the network or a key."""
    embedder = OpenAIEmbedder()
    assert embedder.dim == 1536
    assert isinstance(OpenAIEmbedder.is_available(), bool)
    assert isinstance(SentenceTransformerEmbedder.is_available(), bool)
    assert embedder_capabilities()["openai"]["needs_key"] is True


def test_registry_lookup_and_custom_registration():
    assert isinstance(get_embedder("hashing", dim=32), HashingEmbedder)
    register_embedder("tiny", lambda: HashingEmbedder(dim=8))
    assert get_embedder("tiny").dim == 8
    try:
        get_embedder("does-not-exist")
    except KeyError as exc:
        assert "available" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_sublinear_damping_keeps_repetition_from_dominating():
    e = HashingEmbedder(dim=256)
    once = e.embed_one("filter")
    many = e.embed_one(" ".join(["filter"] * 20))
    unrelated = e.embed_one("warranty period in months")
    # Twenty repetitions must still look like the same word, not a new one.
    assert cosine(once, many) > 0.8
    assert cosine(once, many) > 5 * cosine(once, unrelated)
    assert abs(l2_norm(many) - 1.0) < 1e-9
    assert not math.isnan(cosine(once, many))
