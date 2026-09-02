#!/usr/bin/env python3
"""Measure chunking strategies against each other on the bundled goldset.

    python3 examples/compare_chunking.py

This is the script that turns "structure-aware chunking is better" from an
opinion into a number. Point it at your own corpus and goldset and it will
tell you which strategy wins there, which is not guaranteed to be the same
answer.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (path setup)

from faqbot import PipelineConfig, compare, load_goldset
from faqbot.eval import format_comparison_table

CONFIGS = [
    PipelineConfig(
        chunker="structure",
        chunker_kwargs={"max_tokens": 180, "overlap_sentences": 1},
        label="structure-180",
    ),
    PipelineConfig(
        chunker="sentence",
        chunker_kwargs={"max_tokens": 180, "overlap_sentences": 1},
        label="sentence-180",
    ),
    PipelineConfig(
        chunker="fixed",
        chunker_kwargs={"max_tokens": 120, "overlap": 20},
        label="fixed-120",
    ),
    PipelineConfig(
        chunker="fixed",
        chunker_kwargs={"max_tokens": 60, "overlap": 10},
        label="fixed-60",
    ),
]


def main() -> int:
    goldset = load_goldset(_bootstrap.GOLDSET)
    reports = compare(CONFIGS, goldset, corpus_paths=[_bootstrap.CORPUS])
    print(format_comparison_table(reports))
    print()
    best = max(reports, key=lambda r: (r.content_accuracy, r.mrr))
    print("best expected-content accuracy: %s (%.3f)" % (best.label, best.content_accuracy))
    print("measured on the bundled sample corpus with the offline hashing embedder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
