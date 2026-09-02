"""Chunking: the heading must stay with its body, and a Q must stay with its A.

This is the file that pins down the central claim of the repository. A fixed
window splits a question away from its answer; the structure-aware chunker does
not.
"""

from __future__ import annotations

from faqbot.chunking import (
    FixedTokenChunker,
    SentenceChunker,
    StructureAwareChunker,
    chunk_documents,
    get_chunker,
    parse_sections,
)
from faqbot.types import Document


def test_parse_sections_tracks_the_heading_path(faq_doc):
    sections = parse_sections(faq_doc.text)
    breadcrumbs = [s.breadcrumb for s in sections]
    assert ("Battery",) in breadcrumbs
    assert ("Battery", "How long does the AR-1 battery last?") in breadcrumbs
    assert ("Battery", "How do I replace the filter NW-FILT-02?") in breadcrumbs


def test_question_headings_are_detected():
    sections = parse_sections("## How long does it last?\n\nNinety minutes.\n\n## Overview\n\nText.")
    by_heading = {s.heading: s.is_question for s in sections}
    assert by_heading["How long does it last?"] is True
    assert by_heading["Overview"] is False


def test_fenced_code_hashes_are_not_headings():
    text = "## Setup\n\n```bash\n# not a heading\nrun --now\n```\n\nMore body text here.\n"
    headings = [s.heading for s in parse_sections(text)]
    assert headings == ["", "Setup"] or headings == ["Setup"]
    assert "# not a heading" not in headings


def test_structure_chunker_keeps_heading_with_body(faq_doc):
    chunks = StructureAwareChunker(max_tokens=40).split(faq_doc)
    battery = [c for c in chunks if "90 minutes" in c.text]
    assert len(battery) == 1
    assert "How long does the AR-1 battery last?" in battery[0].text
    assert battery[0].breadcrumb == ("Battery", "How long does the AR-1 battery last?")


def test_structure_chunker_never_splits_a_qa_pair(faq_doc):
    """Every question heading appears in exactly one chunk, with its answer."""
    chunks = StructureAwareChunker(max_tokens=30).split(faq_doc)
    for question, answer_marker in (
        ("How long does the AR-1 battery last?", "90 minutes"),
        ("How do I replace the filter NW-FILT-02?", "Twist the dust bin"),
    ):
        holding = [c for c in chunks if question in c.text]
        assert len(holding) == 1, "question %r landed in %d chunks" % (question, len(holding))
        assert answer_marker in holding[0].text


LONG_ANSWER_DOC = Document(
    doc_id="long",
    source="service.md",
    title="Service",
    text=(
        "## How do I service the main roller?\n\n"
        "Press the two orange tabs on the roller cover and lift it off. "
        "Pull the roller out from the tabbed end first. "
        "Cut away wrapped hair with the blade under the bin lid. "
        "Check both end caps for play before refitting. "
        "Refit the roller until both tabs click. "
        "Run a short clean and listen for a rattle. "
        "A rattle means the roller is not seated. "
        "The roller part number is NW-ROLL-07.\n"
    ),
)


def test_fixed_chunker_separates_the_answer_from_its_question():
    """The baseline failure, asserted so the contrast is not just a claim.

    An answer longer than the window is guaranteed to overflow it. Everything
    after the overflow is a chunk with no question attached, which is exactly
    the chunk that will never be retrieved for the question it answers.
    """
    question = "How do I service the main roller?"
    chunks = FixedTokenChunker(max_tokens=20, overlap=3).split(LONG_ANSWER_DOC)
    orphans = [c for c in chunks if question not in c.text]
    assert orphans, "expected fixed-width chunking to orphan part of the answer"
    assert any("NW-ROLL-07" in c.text for c in orphans)


def test_structure_chunker_keeps_the_whole_answer_with_its_question():
    question = "How do I service the main roller?"
    chunks = StructureAwareChunker(max_tokens=200).split(LONG_ANSWER_DOC)
    assert len(chunks) == 1
    assert question in chunks[0].text
    assert "NW-ROLL-07" in chunks[0].text


def test_structure_chunker_drops_empty_container_headings(faq_doc):
    chunks = StructureAwareChunker(max_tokens=40).split(faq_doc)
    assert all(c.text.strip() != "Battery" for c in chunks)


def test_long_section_splits_but_every_piece_keeps_its_heading():
    body = " ".join("Sentence number %d explains a detail of the procedure." % i for i in range(40))
    doc = Document(doc_id="d", source="s", title="T", text="## How do I service it?\n\n%s" % body)
    chunks = StructureAwareChunker(max_tokens=30, hard_limit_factor=1.5).split(doc)
    assert len(chunks) > 1
    assert all("How do I service it?" in c.text for c in chunks)
    assert all(c.breadcrumb[-1] == "How do I service it?" for c in chunks)


def test_chunk_ids_are_deterministic(faq_doc):
    a = StructureAwareChunker(max_tokens=40).split(faq_doc)
    b = StructureAwareChunker(max_tokens=40).split(faq_doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len(set(c.chunk_id for c in a)) == len(a)


def test_sentence_chunker_never_cuts_mid_sentence(faq_doc):
    chunks = SentenceChunker(max_tokens=25, overlap_sentences=1).split(faq_doc)
    assert chunks
    for chunk in chunks:
        assert not chunk.text.endswith(" on")
        assert not chunk.text.endswith(",")


def test_sentence_overlap_repeats_the_boundary_sentence():
    text = " ".join("This is sentence number %d of the document." % i for i in range(12))
    doc = Document(doc_id="d", source="s", title="T", text=text)
    chunks = SentenceChunker(max_tokens=20, overlap_sentences=1).split(doc)
    assert len(chunks) > 1
    tail = chunks[0].text.split("\n")[-1]
    assert tail in chunks[1].text


def test_fixed_chunker_rejects_bad_overlap():
    try:
        FixedTokenChunker(max_tokens=10, overlap=10)
    except ValueError:
        pass
    else:  # pragma: no cover - the constructor must reject this
        raise AssertionError("overlap >= max_tokens should raise")


def test_registry_returns_the_right_class():
    assert isinstance(get_chunker("structure"), StructureAwareChunker)
    assert isinstance(get_chunker("fixed", max_tokens=50), FixedTokenChunker)
    try:
        get_chunker("nope")
    except KeyError as exc:
        assert "available" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown chunker should raise KeyError")


def test_every_structure_chunk_carries_its_heading(corpus_dir):
    """On the real corpus every structure chunk keeps its own heading.

    The comparison is the point. Fixed-width chunking starts most of its chunks
    in the middle of a sentence, with no heading and no subject; those chunks
    are unretrievable by the question they answer, which is why the eval
    harness scores fixed chunking lower on content accuracy.
    """
    from faqbot.ingest import load_directory

    docs = load_directory(corpus_dir)
    assert len(docs) > 10

    structure = [c for d in docs for c in StructureAwareChunker(max_tokens=180).split(d)]
    assert structure
    for chunk in structure:
        assert chunk.breadcrumb
        assert chunk.breadcrumb[-1] in chunk.text
    assert not [c for c in structure if c.text[:1].islower()]

    fixed = [c for d in docs for c in FixedTokenChunker(max_tokens=60, overlap=10).split(d)]
    mid_sentence = [c for c in fixed if c.text[:1].islower()]
    assert len(mid_sentence) > len(fixed) // 3


def test_chunk_documents_preserves_document_order(faq_doc):
    other = Document(doc_id="d2", source="b.md", title="Other", text="## Q?\n\nAn answer body here.")
    chunks = chunk_documents([faq_doc, other], StructureAwareChunker(max_tokens=60))
    doc_ids = [c.doc_id for c in chunks]
    assert doc_ids.index("doc1") < doc_ids.index("d2")
