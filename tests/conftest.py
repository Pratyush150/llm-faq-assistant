"""Shared fixtures. Everything here is offline and deterministic."""

from __future__ import annotations

import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from faqbot.types import Chunk, Document  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CORPUS_DIR = os.path.join(REPO_ROOT, "data", "corpus")
GOLDSET_PATH = os.path.join(REPO_ROOT, "data", "goldset.json")
ADVERSARIAL_DIR = os.path.join(REPO_ROOT, "data", "adversarial")

FAQ_MARKDOWN = """# Battery

Everything about power.

## How long does the AR-1 battery last?

Up to 90 minutes in Eco mode and about 45 minutes in Turbo mode. Runtime drops
on deep carpet because the brush motor draws more current.

## How do I replace the filter NW-FILT-02?

Twist the dust bin counter-clockwise, lift the filter out, rinse it under cold
water and let it dry for 24 hours before refitting.
"""


@pytest.fixture()
def faq_doc() -> Document:
    return Document(doc_id="doc1", source="faq.md", title="Battery", text=FAQ_MARKDOWN)


@pytest.fixture()
def toy_chunks() -> List[Chunk]:
    """Three tiny chunks with hand-countable token statistics."""
    return [
        Chunk(chunk_id="c1", doc_id="d1", text="the quick brown fox jumps"),
        Chunk(chunk_id="c2", doc_id="d2", text="the lazy brown dog sleeps"),
        Chunk(chunk_id="c3", doc_id="d3", text="a fox and a dog"),
    ]


@pytest.fixture(scope="session")
def corpus_chunks():
    """Chunks of the real bundled corpus, for tests that need a real vocabulary."""
    from faqbot.chunking import StructureAwareChunker, chunk_documents
    from faqbot.ingest import load_directory

    if not os.path.isdir(CORPUS_DIR):
        pytest.skip("bundled corpus is not present")
    return chunk_documents(load_directory(CORPUS_DIR), StructureAwareChunker())


@pytest.fixture()
def corpus_dir() -> str:
    if not os.path.isdir(CORPUS_DIR):
        pytest.skip("bundled corpus is not present")
    return CORPUS_DIR
