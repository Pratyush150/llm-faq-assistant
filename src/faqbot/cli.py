"""Command line interface: ``ingest``, ``ask``, ``serve``, ``eval``, ``--demo``.

``--demo`` is the one that matters. It indexes the bundled sample corpus,
answers a scripted set of questions covering every guardrail, and prints the
evaluation report, with no arguments, no network and no keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .embedding import embedder_capabilities
from .eval import compare, format_comparison_table, load_goldset, run_eval
from .guardrails import GuardrailConfig
from .pipeline import FAQPipeline, PipelineConfig

__all__ = ["main", "build_parser"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
def _friendly(path: str) -> str:
    """Shortest readable spelling of a path, for output that is pasted around."""
    try:
        relative = os.path.relpath(path)
    except ValueError:  # pragma: no cover - different drives on Windows
        return path
    return relative if not relative.startswith("..") and len(relative) < len(path) else path


DEFAULT_CORPUS = _friendly(os.path.join(_REPO_ROOT, "data", "corpus"))
DEFAULT_GOLDSET = _friendly(os.path.join(_REPO_ROOT, "data", "goldset.json"))
DEFAULT_ADVERSARIAL = _friendly(os.path.join(_REPO_ROOT, "data", "adversarial"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faqbot",
        description="Retrieval-grounded FAQ assistant with citations, refusal "
        "guardrails and an evaluation harness. Runs offline by default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  faqbot --demo\n"
            "  faqbot ingest data/corpus --save index/faq.json\n"
            "  faqbot ask 'How long does the AR-1 battery last?' --corpus data/corpus\n"
            "  faqbot serve --corpus data/corpus --port 8080\n"
            "  faqbot eval data/goldset.json --compare\n"
        ),
    )
    parser.add_argument("--version", action="version", version="faqbot %s" % __version__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="index the bundled sample corpus and run a scripted end-to-end demo",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", action="append", default=None, help="file or directory to index (repeatable)")
    common.add_argument("--index", default=None, help="load a persisted index instead of ingesting")
    common.add_argument("--chunker", default="structure", choices=["structure", "sentence", "fixed"])
    common.add_argument("--max-tokens", type=int, default=180, help="chunk size budget in tokens")
    common.add_argument("--embedder", default="hashing", help="embedder name (default: offline hashing)")
    common.add_argument("--dim", type=int, default=256, help="hashing embedder dimension")
    common.add_argument("--top-k", type=int, default=5)
    common.add_argument("--no-rerank", action="store_true")
    common.add_argument("--dense-only", action="store_true", help="disable BM25 (dense retrieval only)")
    common.add_argument("--sparse-only", action="store_true", help="disable vectors (BM25 only)")
    common.add_argument("--min-score", type=float, default=None, help="override the refusal confidence floor")

    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", parents=[common], help="index documents and optionally save the index")
    p_ingest.add_argument("paths", nargs="+")
    p_ingest.add_argument("--save", default=None, help="write the index to this path (.json or .npz)")

    p_ask = sub.add_parser("ask", parents=[common], help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="print the full answer as JSON")
    p_ask.add_argument("--show-retrieval", action="store_true", help="print the retrieved chunks and scores")

    p_serve = sub.add_parser("serve", parents=[common], help="run the HTTP API and chat UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--goldset", default=None, help="goldset used by the /eval endpoint")
    p_serve.add_argument("--no-ingest", action="store_true", help="disable the /ingest endpoint")

    p_eval = sub.add_parser("eval", parents=[common], help="run a goldset and print metrics")
    p_eval.add_argument("goldset", nargs="?", default=None)
    p_eval.add_argument("--compare", action="store_true", help="compare several chunking and retrieval configurations")
    p_eval.add_argument("--json", default=None, help="write the full report to this JSON file")
    p_eval.add_argument("--failures", action="store_true", help="print the questions that failed")

    sub.add_parser("capabilities", help="report which embedders and optional plugins are usable here")
    return parser


def _config_from_args(args: argparse.Namespace, label: str = "cli") -> PipelineConfig:
    if args.chunker == "fixed":
        chunker_kwargs: Dict[str, Any] = {"max_tokens": args.max_tokens, "overlap": max(1, args.max_tokens // 8)}
    else:
        chunker_kwargs = {"max_tokens": args.max_tokens, "overlap_sentences": 1}
    guards = GuardrailConfig()
    if args.min_score is not None:
        guards.min_top_score = args.min_score
    return PipelineConfig(
        chunker=args.chunker,
        chunker_kwargs=chunker_kwargs,
        embedder=args.embedder,
        embedder_kwargs={"dim": args.dim} if args.embedder == "hashing" else {},
        top_k=args.top_k,
        reranker="none" if args.no_rerank else "feature",
        use_dense=not args.sparse_only,
        use_sparse=not args.dense_only,
        guardrails=guards,
        label=label,
    )


def _build_pipeline(args: argparse.Namespace, paths: Optional[Sequence[str]] = None) -> FAQPipeline:
    pipe = FAQPipeline(_config_from_args(args))
    if args.index:
        pipe.load(args.index)
        return pipe
    sources = list(paths or args.corpus or [DEFAULT_CORPUS])
    pipe.ingest(sources)
    return pipe


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    pipe = FAQPipeline(_config_from_args(args, "ingest"))
    stats = pipe.ingest(args.paths)
    print(
        "indexed %d documents -> %d chunks (%s chunker, %s embedder, dim %d)"
        % (stats["documents"], stats["chunks_total"], pipe.chunker.name, pipe.embedder.name, pipe.embedder.dim)
    )
    if stats["chunks_replaced"]:
        print("  replaced %d chunks from previously indexed documents" % stats["chunks_replaced"])
    if args.save:
        path = pipe.save(args.save)
        print("  index written to %s (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipe = _build_pipeline(args)
    if args.show_retrieval:
        print("retrieved:")
        for sc in pipe.retrieve(args.question):
            print(
                "  %.3f  mq=%.3f  %s"
                % (sc.score, sc.components.get("match_quality", 0.0), sc.chunk.label)
            )
        print()
    answer = pipe.ask(args.question)
    print(answer.to_json() if args.json else answer.render())
    return 2 if answer.refused else 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    pipe = _build_pipeline(args)
    goldset = args.goldset or (DEFAULT_GOLDSET if os.path.exists(DEFAULT_GOLDSET) else None)
    serve(
        pipe,
        host=args.host,
        port=args.port,
        goldset_path=goldset,
        allow_ingest=not args.no_ingest,
    )
    return 0


_COMPARE_CONFIGS: List[Dict[str, Any]] = [
    {"label": "structure+hybrid", "chunker": "structure", "chunker_kwargs": {"max_tokens": 180, "overlap_sentences": 1}},
    {"label": "sentence+hybrid", "chunker": "sentence", "chunker_kwargs": {"max_tokens": 180, "overlap_sentences": 1}},
    {"label": "fixed120+hybrid", "chunker": "fixed", "chunker_kwargs": {"max_tokens": 120, "overlap": 20}},
    {"label": "fixed60+hybrid", "chunker": "fixed", "chunker_kwargs": {"max_tokens": 60, "overlap": 10}},
    {"label": "structure+dense", "chunker": "structure", "chunker_kwargs": {"max_tokens": 180, "overlap_sentences": 1}, "use_sparse": False},
    {"label": "structure+bm25", "chunker": "structure", "chunker_kwargs": {"max_tokens": 180, "overlap_sentences": 1}, "use_dense": False},
    {"label": "structure+norerank", "chunker": "structure", "chunker_kwargs": {"max_tokens": 180, "overlap_sentences": 1}, "reranker": "none"},
]


def cmd_eval(args: argparse.Namespace) -> int:
    path = args.goldset or DEFAULT_GOLDSET
    if not os.path.exists(path):
        print("no goldset at %s" % path, file=sys.stderr)
        return 1
    goldset = load_goldset(path)
    corpus = list(args.corpus or ([goldset.corpus] if goldset.corpus else [DEFAULT_CORPUS]))

    if args.compare:
        configs = []
        for spec in _COMPARE_CONFIGS:
            cfg = _config_from_args(args, str(spec["label"]))
            cfg.chunker = str(spec["chunker"])
            cfg.chunker_kwargs = dict(spec["chunker_kwargs"])
            cfg.use_dense = bool(spec.get("use_dense", True))
            cfg.use_sparse = bool(spec.get("use_sparse", True))
            cfg.reranker = str(spec.get("reranker", "feature"))
            configs.append(cfg)
        reports = compare(configs, goldset, corpus_paths=corpus)
        print(format_comparison_table(reports))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump([r.to_dict() for r in reports], fh, indent=2)
            print("\nfull report written to %s" % args.json)
        return 0

    pipe = _build_pipeline(args, corpus)
    report = run_eval(pipe, goldset)
    print(report.render())
    if args.failures:
        bad = report.failures()
        print("\n%d question(s) to look at:" % len(bad))
        for r in bad:
            print("  [%s] %s" % (r.qid, r.question))
            print("      refused=%s reason=%s rank=%s" % (r.refused, r.refusal_reason, r.first_relevant_rank))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print("\nfull report written to %s" % args.json)
    return 0


def cmd_capabilities(_args: argparse.Namespace) -> int:
    print("faqbot %s" % __version__)
    print("\nembedders:")
    for name, info in embedder_capabilities().items():
        state = "available" if info["available"] else "not available"
        extra = []
        if info["offline"]:
            extra.append("offline")
        if info["needs_key"]:
            extra.append("needs API key")
        print("  %-22s %-14s %s" % (name, state, ", ".join(extra)))
    print("\nOptional features:")
    from .rerank import CrossEncoderReranker
    from .store import VectorStore

    print("  npz index persistence  %s" % ("available" if VectorStore.npz_available() else "needs numpy"))
    print("  cross-encoder rerank   %s" % ("available" if CrossEncoderReranker.is_available() else "needs sentence-transformers"))
    try:
        import importlib.util

        has_yaml = importlib.util.find_spec("yaml") is not None
    except Exception:
        has_yaml = False
    print("  YAML goldsets          %s" % ("available" if has_yaml else "needs PyYAML"))
    print("\nThe default path (hashing embedder + extractive answerer) needs none of these.")
    return 0


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

_DEMO_QUESTIONS = [
    ("How long does the AR-1 battery last on one charge?", "a straightforward lookup"),
    ("Which filter part number should I buy?", "an identifier BM25 finds and embeddings miss"),
    ("What does error code E03 mean?", "a rare token that has to beat nine similar chunks"),
    ("What is the best recipe for sourdough bread?", "out of domain: must refuse"),
    ("Does the AR-1 support Matter over Thread?", "in-vocabulary but unanswerable: must refuse"),
]

_DEMO_FOLLOWUP = [
    "How long does the AR-1 battery last?",
    "Does it work on carpet?",
    "What about the Pro version?",
]


def _clip(text: str, width: int = 200) -> str:
    """Shorten to a whole word, so demo output never breaks mid-word."""
    if len(text) <= width:
        return text
    return text[:width].rsplit(" ", 1)[0] + " ..."


def _rule(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def cmd_demo() -> int:
    if not os.path.isdir(DEFAULT_CORPUS):
        print("bundled corpus missing at %s" % DEFAULT_CORPUS, file=sys.stderr)
        return 1

    _rule("1. INDEX THE BUNDLED CORPUS (offline, no API key)")
    pipe = FAQPipeline(PipelineConfig(label="demo"))
    stats = pipe.ingest([DEFAULT_CORPUS])
    print(
        "%d source documents -> %d chunks | %s chunker | %s embedder (dim %d) | %d vocabulary terms"
        % (
            stats["documents"],
            stats["chunks_total"],
            pipe.chunker.name,
            pipe.embedder.name,
            pipe.embedder.dim,
            len(pipe.guards.vocab),
        )
    )

    _rule("2. ANSWERING, WITH CITATIONS AND REFUSALS")
    for question, why in _DEMO_QUESTIONS:
        print("\n> %s" % question)
        print("  (%s)" % why)
        print()
        answer = pipe.ask(question)
        print("\n".join("  " + line for line in answer.render().split("\n")))

    _rule("3. FOLLOW-UP QUESTIONS (query rewriting from conversation memory)")
    for question in _DEMO_FOLLOWUP:
        answer = pipe.ask(question, session_id="demo")
        print("\n> %s" % question)
        if answer.rewritten_question != question:
            print("  rewritten as: %s" % answer.rewritten_question)
        print("  %s" % _clip(answer.text))

    _rule("4. PROMPT INJECTION IN A RETRIEVED DOCUMENT")
    if os.path.isdir(DEFAULT_ADVERSARIAL):
        adv = FAQPipeline(PipelineConfig(label="demo-adversarial"))
        adv.ingest([DEFAULT_CORPUS, DEFAULT_ADVERSARIAL])
        question = "What do owners say about AR-1 battery life?"
        answer = adv.ask(question)
        hits = answer.diagnostics.get("injection_hits", [])
        print("\n> %s" % question)
        print("  injection patterns detected in retrieved content: %d" % len(hits))
        for hit in hits[:4]:
            print("    %-22s %s" % (hit["kind"], hit["text"][:60]))
        print("  the instruction was neutralised, not obeyed. Answer:")
        print("\n".join("    " + line for line in answer.render().split("\n")))
    else:
        print("  (adversarial sample not bundled)")

    _rule("5. EVALUATION ON THE BUNDLED GOLDSET")
    if os.path.exists(DEFAULT_GOLDSET):
        goldset = load_goldset(DEFAULT_GOLDSET)
        report = run_eval(pipe, goldset)
        print(report.render())
    else:
        print("  (goldset not bundled)")

    print()
    print("Next: faqbot serve --corpus data/corpus   ->  browser UI on http://127.0.0.1:8080")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.demo:
        return cmd_demo()
    if not args.command:
        parser.print_help()
        return 1
    handlers = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "serve": cmd_serve,
        "eval": cmd_eval,
        "capabilities": cmd_capabilities,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
