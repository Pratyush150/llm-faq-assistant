#!/usr/bin/env python3
"""Smallest useful program: index the sample corpus and ask three questions.

    python3 examples/run_demo.py

No API key, no network, no optional dependencies.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (path setup)

from faqbot import FAQPipeline

QUESTIONS = [
    "How long does the AR-1 battery last on one charge?",
    "Which filter part number does the AR-1 use?",
    "How much does a return flight to Lisbon cost?",
]


def main() -> int:
    pipe = FAQPipeline()
    stats = pipe.ingest([_bootstrap.CORPUS])
    print("indexed %d documents into %d chunks\n" % (stats["documents"], stats["chunks_total"]))

    for question in QUESTIONS:
        print("Q: %s" % question)
        print(pipe.ask(question).render())
        print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
