"""Tokenisation and sentence splitting.

These are the primitives BM25, the embedder, the reranker and the grounding
check all share. When they drift, every metric moves for no visible reason.
"""

from __future__ import annotations

import pytest

from faqbot.textutil import (
    coverage,
    jaccard,
    light_stem,
    locate_span,
    longest_common_phrase,
    normalize_text,
    normalize_whitespace,
    split_sentences,
    tokenize,
)


def test_identifiers_survive_tokenisation():
    """Splitting NW-FILT-02 into three tokens is how keyword search 'breaks'."""
    assert tokenize("Order NW-FILT-02 and NW-BATT-32") == ["order", "nw-filt-02", "and", "nw-batt-32"]
    assert "ar-1" in tokenize("The AR-1 Pro")
    assert "v2.3" in tokenize("firmware v2.3 is current")


def test_stopwords_are_dropped_only_when_asked():
    assert "the" in tokenize("the filter")
    assert "the" not in tokenize("the filter", drop_stopwords=True)


def test_normalisation_is_case_and_whitespace_insensitive():
    assert normalize_text("  The   AR-1\nPro ") == "the ar-1 pro"


def test_whitespace_normalisation_keeps_markdown_structure():
    text = "# Title\n\n\n\nSome   body    text.\n\n- one\n- two\n\n\n"
    out = normalize_whitespace(text)
    assert out.startswith("# Title")
    assert "\n\n\n" not in out
    assert "Some  body  text." in out
    assert out.endswith("- two")


def test_column_gaps_survive_normalisation_and_split_as_table_rows():
    """A plain-text spec sheet must not become one 300-word 'sentence'."""
    table = "Runtime               up to 90 minutes\nCharge time           about 240 minutes\n"
    out = normalize_whitespace(table)
    assert "Runtime  up to 90 minutes" in out
    assert split_sentences(out) == ["Runtime  up to 90 minutes", "Charge time  about 240 minutes"]


def test_sentences_are_joined_across_hard_wrapped_lines():
    """The bug this guards: citation quotes that stop mid-clause."""
    text = "Runtime is up to 90 minutes in Eco mode and about 45 minutes in\nTurbo mode. It drops on carpet."
    sentences = split_sentences(text)
    assert sentences[0].endswith("Turbo mode.")
    assert len(sentences) == 2


def test_abbreviations_do_not_end_a_sentence():
    assert split_sentences("This is Dr. Smith speaking. Yes.") == [
        "This is Dr. Smith speaking.",
        "Yes.",
    ]


def test_headings_and_list_items_are_their_own_sentences():
    sentences = split_sentences("# Title\n\nBody text here.\n\n- first item\n- second item")
    assert "# Title" in sentences
    assert "- first item" in sentences
    assert "Body text here." in sentences


def test_light_stem_handles_plurals_without_mangling_identifiers():
    assert light_stem("codes") == "code"
    assert light_stem("batteries") == "battery"
    assert light_stem("classes") == "class"
    assert light_stem("gas") == "gas"
    assert light_stem("nw-filt-02") == "nw-filt-02"
    assert light_stem("mopping") == "mopping"


def test_coverage_is_asymmetric_and_stemmed():
    assert coverage(tokenize("error code"), tokenize("Error codes are listed here")) == pytest.approx(1.0)
    assert coverage(tokenize("error code e03"), tokenize("Error codes")) == pytest.approx(2.0 / 3.0)
    assert coverage([], ["anything"]) == pytest.approx(0.0)


def test_jaccard_is_symmetric():
    a, b = ["x", "y"], ["y", "z"]
    assert jaccard(a, b) == pytest.approx(jaccard(b, a))
    assert jaccard(a, b) == pytest.approx(1.0 / 3.0)


def test_longest_common_phrase_finds_a_contiguous_run():
    length, phrase = longest_common_phrase(
        "runtime is 90 minutes on eco", "the runtime is 90 minutes on eco mode"
    )
    assert length == 6
    assert phrase == "runtime is 90 minutes on eco"


def test_longest_common_phrase_respects_the_minimum():
    length, phrase = longest_common_phrase("alpha", "alpha beta", min_tokens=2)
    assert length == 1
    assert phrase == ""


def test_locate_span_tolerates_a_line_break():
    haystack = "Runtime is up to 90 minutes in Eco mode and about\n45 minutes in Turbo mode."
    needle = "Runtime is up to 90 minutes in Eco mode and about 45 minutes in Turbo mode."
    start, end = locate_span(haystack, needle)
    assert (start, end) == (0, len(haystack))


def test_locate_span_returns_a_degenerate_span_when_absent():
    assert locate_span("abc", "not here", 2) == (2, 2)
