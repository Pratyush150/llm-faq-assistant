"""Ingest: format coverage, normalisation and idempotent document ids."""

from __future__ import annotations

import os

from faqbot.ingest import (
    IngestConfig,
    compute_doc_id,
    html_to_text,
    load_directory,
    load_file,
    load_paths,
    read_text_with_fallback,
)


def _write(tmp_path, name: str, content: str, encoding: str = "utf-8") -> str:
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding=encoding) as fh:
        fh.write(content)
    return path


def test_markdown_document_is_loaded_and_titled(tmp_path):
    path = _write(tmp_path, "a.md", "# Battery life\n\nUp to 90 minutes in Eco mode.\n")
    docs = load_file(path)
    assert len(docs) == 1
    assert docs[0].title == "Battery life"
    assert "90 minutes" in docs[0].text
    assert docs[0].meta["kind"] == "markdown"


def test_doc_id_is_content_addressed_and_stable(tmp_path):
    path = _write(tmp_path, "a.md", "# Battery\n\nUp to 90 minutes.\n")
    first = load_file(path)[0]
    second = load_file(path)[0]
    assert first.doc_id == second.doc_id
    assert len(first.doc_id) == 16


def test_doc_id_ignores_whitespace_churn():
    a = compute_doc_id("Up to 90 minutes in Eco mode.", "faq.md")
    b = compute_doc_id("Up   to 90 minutes\nin Eco mode.", "faq.md")
    assert a == b


def test_doc_id_changes_with_source():
    assert compute_doc_id("same text", "one.md") != compute_doc_id("same text", "two.md")


def test_reingesting_the_same_paths_is_idempotent(tmp_path):
    _write(tmp_path, "a.md", "# One\n\nThe AR-1 runs for 90 minutes.\n")
    _write(tmp_path, "b.md", "# Two\n\nThe filter is NW-FILT-02.\n")
    once = load_paths([str(tmp_path)])
    twice = load_paths([str(tmp_path), str(tmp_path)])
    assert len(once) == 2
    assert len(twice) == 2
    assert sorted(d.doc_id for d in once) == sorted(d.doc_id for d in twice)


def test_encoding_fallback_reads_cp1252(tmp_path):
    """A cp1252 export mislabelled as UTF-8 must not abort the ingest."""
    path = os.path.join(str(tmp_path), "legacy.txt")
    with open(path, "wb") as fh:
        # 0x92 is a cp1252 right single quote; it is invalid UTF-8.
        fh.write(b"Don\x92t run the AR-1 over cables.")
    text = read_text_with_fallback(path)
    assert "run the AR-1 over cables" in text
    assert "\u2019" in text  # decoded as a cp1252 smart quote, not mojibake


def test_html_is_converted_to_headed_text_and_scripts_dropped():
    html = (
        "<html><head><title>Support</title></head><body>"
        "<nav>Home &gt; Docs</nav><h1>Filters</h1>"
        "<p>Use NW-FILT-02.</p><script>evil()</script></body></html>"
    )
    text, title = html_to_text(html, ("script", "style", "nav"))
    assert title == "Support"
    assert text.startswith("# Filters")
    assert "NW-FILT-02" in text
    assert "evil" not in text
    assert "Home" not in text


def test_csv_faq_pairs_become_one_document_per_row(tmp_path):
    path = _write(
        tmp_path,
        "faq.csv",
        "question,answer,category\n"
        "How long is the warranty?,24 months on the robot.,warranty\n"
        "Which filter?,NW-FILT-02.,parts\n",
    )
    docs = load_file(path)
    assert len(docs) == 2
    assert docs[0].text.startswith("## How long is the warranty?")
    assert "24 months" in docs[0].text
    assert docs[0].meta["category"] == "warranty"
    assert docs[0].meta["kind"] == "faq_pair"


def test_json_faq_pairs_are_rendered_as_qa_markdown(tmp_path):
    path = _write(
        tmp_path,
        "faq.json",
        '{"faqs": [{"question": "Does it mop?", "answer": "Only the Pro model mops."}]}',
    )
    docs = load_file(path)
    assert len(docs) == 1
    assert docs[0].text == "## Does it mop?\n\nOnly the Pro model mops."


def test_boilerplate_lines_are_stripped(tmp_path):
    path = _write(
        tmp_path,
        "page.md",
        "# Returns\n\nAll rights reserved\n\nReturns are accepted within 30 days.\n"
        "Cookie policy applies to this site\n",
    )
    doc = load_file(path)[0]
    assert "30 days" in doc.text
    assert "All rights reserved" not in doc.text
    assert "Cookie policy" not in doc.text


def test_tiny_documents_are_skipped(tmp_path):
    path = _write(tmp_path, "stub.md", "hi\n")
    assert load_file(path, IngestConfig(min_chars=20)) == []


def test_load_directory_is_sorted_and_recursive(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "sub"))
    _write(tmp_path, "b.md", "# B\n\nThe AR-1 charges in 240 minutes.\n")
    _write(tmp_path, "a.md", "# A\n\nThe AR-1 runs for 90 minutes.\n")
    with open(os.path.join(str(tmp_path), "sub", "c.txt"), "w", encoding="utf-8") as fh:
        fh.write("The filter part number is NW-FILT-02.\n")
    docs = load_directory(str(tmp_path))
    assert [os.path.basename(d.source) for d in docs] == ["a.md", "b.md", "c.txt"]


def test_unsupported_suffixes_are_ignored(tmp_path):
    _write(tmp_path, "keep.md", "# Keep\n\nThis document is indexed fine.\n")
    _write(tmp_path, "skip.bin", "binary-ish content that should not be indexed")
    docs = load_directory(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].source.endswith("keep.md")
