#!/usr/bin/env python3
"""Index your own documents, persist the index, reload it and query it.

    python3 examples/custom_corpus.py /path/to/your/docs

Supported inputs: .md, .txt, .html, .csv and .json. With no argument it writes
a two-file throwaway corpus to a temporary directory so the script still runs.
"""

from __future__ import annotations

import os
import sys
import tempfile

import _bootstrap  # noqa: F401  (path setup)

from faqbot import FAQPipeline, PipelineConfig

SAMPLE = {
    "hours.md": (
        "# Opening hours\n\n"
        "## When is the workshop open?\n\n"
        "Weekdays from 08:00 to 18:00. Closed on public holidays.\n"
    ),
    "returns.csv": (
        "question,answer\n"
        "How do I return an item?,"
        '"Bring the item and the receipt within 30 days. Refunds take 14 days."\n'
    ),
}


def _sample_corpus() -> str:
    root = tempfile.mkdtemp(prefix="faqbot-sample-")
    for name, body in SAMPLE.items():
        with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    return root


def main(argv: list) -> int:
    corpus = argv[1] if len(argv) > 1 else _sample_corpus()
    index_path = os.path.join(tempfile.mkdtemp(prefix="faqbot-index-"), "index.json")

    pipe = FAQPipeline(PipelineConfig(label="custom"))
    stats = pipe.ingest([corpus])
    print("corpus     : %s" % corpus)
    print("indexed    : %d documents -> %d chunks" % (stats["documents"], stats["chunks_total"]))

    # Re-ingesting the same corpus replaces chunks instead of duplicating them.
    again = pipe.ingest([corpus])
    print("re-ingest  : %d added, %d replaced, %d total"
          % (again["chunks_added"], again["chunks_replaced"], again["chunks_total"]))

    pipe.save(index_path)
    print("saved      : %s (%.1f KB)" % (index_path, os.path.getsize(index_path) / 1024.0))

    reloaded = FAQPipeline(PipelineConfig(label="custom")).load(index_path)
    print("reloaded   : %d chunks\n" % len(reloaded.store))

    for question in ("When is the workshop open?", "How do I return an item?"):
        print("Q: %s" % question)
        print(reloaded.ask(question).render())
        print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
