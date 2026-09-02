"""Document loading and normalisation.

Supported inputs: ``.md``, ``.txt``, ``.html``/``.htm``, ``.csv`` (question /
answer columns) and ``.json`` (FAQ pairs or a list of documents).

Two properties matter more than the format list:

1. **Deterministic ids.** ``doc_id`` is a hash of the normalised text plus the
   source label. Re-ingesting the same file produces the same id, so an index
   rebuild is an upsert rather than a duplicate. Without this, a nightly
   re-crawl silently triples the corpus and retrieval starts returning three
   copies of the same paragraph, crowding out the answer.

2. **CSV/JSON FAQ pairs are rendered as markdown Q/A blocks**, question as an
   ``##`` heading and answer as the body. That is not cosmetic: it lets the
   structure-aware chunker treat the pair as one atomic unit downstream, which
   is the difference between retrieving an answer and retrieving a question.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .textutil import normalize_text, normalize_whitespace
from .types import Document

__all__ = [
    "IngestConfig",
    "DEFAULT_BOILERPLATE",
    "SUPPORTED_SUFFIXES",
    "compute_doc_id",
    "load_file",
    "load_paths",
    "load_directory",
    "html_to_text",
    "read_text_with_fallback",
]

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".text", ".html", ".htm", ".csv", ".json")

# Lines that are navigation or legal furniture on almost every documentation
# page. They add tokens, dilute BM25 term statistics and never answer anything.
DEFAULT_BOILERPLATE = (
    r"^\s*(all rights reserved|copyright\s*(\(c\)|©)?\s*\d{4}.*)$",
    r"^\s*(cookie (policy|notice|settings)|accept (all )?cookies)\b.*$",
    r"^\s*(skip to (main )?content|back to top|print this page)\s*$",
    r"^\s*(home|menu|navigation|breadcrumbs?)\s*[>|/]\s*.*$",
    r"^\s*(share (on|this)|follow us on)\b.*$",
    r"^\s*(page \d+ of \d+)\s*$",
)

_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


@dataclass
class IngestConfig:
    """Knobs for loading.

    Attributes:
        boilerplate_patterns: Regexes (case-insensitive, per line) whose matches
            are dropped.
        min_chars: Documents shorter than this after normalisation are skipped.
            Guards against indexing empty nav pages that then rank on
            everything because their term counts are tiny.
        strip_html_tags: Tags whose *content* is discarded entirely.
        question_fields / answer_fields: Accepted column or key names for
            tabular FAQ sources, checked case-insensitively.
        extra_meta: Merged into every produced document's metadata.
    """

    boilerplate_patterns: Sequence[str] = DEFAULT_BOILERPLATE
    min_chars: int = 20
    strip_html_tags: Sequence[str] = ("script", "style", "nav", "footer", "noscript", "svg")
    question_fields: Sequence[str] = ("question", "q", "title", "prompt")
    answer_fields: Sequence[str] = ("answer", "a", "response", "body", "text", "content")
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    def compiled_boilerplate(self) -> List["re.Pattern[str]"]:
        return [re.compile(p, re.IGNORECASE) for p in self.boilerplate_patterns]


def read_text_with_fallback(path: str, encodings: Sequence[str] = _ENCODINGS) -> str:
    """Read a text file, trying a small ladder of encodings.

    Real customer corpora are exported from spreadsheets and CMSes and are full
    of cp1252 smart quotes mislabelled as UTF-8. A hard ``UnicodeDecodeError``
    on document 4000 of 4000 is a bad way to end an ingest, so the last rung of
    the ladder (``latin-1``) cannot fail.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    for enc in encodings:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace")


class _TextExtractor(HTMLParser):
    """Stdlib HTML to text. No bs4, no lxml, no network."""

    _BLOCK = {
        "p", "div", "section", "article", "li", "tr", "br", "table", "ul", "ol",
        "blockquote", "pre", "hr", "header", "main", "aside", "dl", "dt", "dd",
    }
    _HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    def __init__(self, drop_tags: Sequence[str]) -> None:
        super().__init__(convert_charrefs=True)
        self._drop = {t.lower() for t in drop_tags}
        self._skip_depth = 0
        self._parts: List[str] = []
        self._pending_heading: Optional[int] = None
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag = tag.lower()
        if tag in self._drop:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag in self._HEADINGS:
            self._parts.append("\n\n")
            self._pending_heading = self._HEADINGS[tag]
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._drop:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in self._HEADINGS or tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        text = data.strip()
        if not text:
            if self._parts and not self._parts[-1].endswith((" ", "\n")):
                self._parts.append(" ")
            return
        if self._pending_heading is not None:
            self._parts.append("#" * self._pending_heading + " " + text + "\n")
            self._pending_heading = None
        else:
            self._parts.append(text + " ")

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(html: str, drop_tags: Sequence[str] = ("script", "style")) -> "tuple[str, str]":
    """Convert HTML to markdown-ish text. Returns ``(text, title)``.

    Headings become ``#`` lines so that the structure-aware chunker can use the
    page's own outline instead of guessing at fixed offsets.
    """
    parser = _TextExtractor(drop_tags)
    parser.feed(html)
    parser.close()
    return normalize_whitespace(parser.text()), parser.title.strip()


def compute_doc_id(text: str, source: str = "") -> str:
    """Stable 16-hex-char content id.

    Hashing the *normalised* text means whitespace churn, a re-export with
    different line endings, or a trailing-newline change do not create a new
    document. Hashing the source alongside it means the same boilerplate answer
    living at two URLs stays two documents, which is what you want when the
    citation has to point somewhere real.
    """
    h = hashlib.sha256()
    h.update(normalize_text(text).encode("utf-8"))
    h.update(b"\x00")
    h.update(source.encode("utf-8"))
    return h.hexdigest()[:16]


def _strip_boilerplate(text: str, patterns: Sequence["re.Pattern[str]"]) -> str:
    if not patterns:
        return text
    kept = [ln for ln in text.split("\n") if not any(p.match(ln) for p in patterns)]
    return "\n".join(kept)


def _first_heading(text: str) -> str:
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:80]
    return ""


def _pick(row: Dict[str, Any], names: Sequence[str]) -> Optional[str]:
    lowered = {str(k).strip().casefold(): v for k, v in row.items() if k is not None}
    for name in names:
        val = lowered.get(name.casefold())
        if val is not None and str(val).strip():
            return str(val)
    return None


def _make_doc(
    text: str,
    source: str,
    title: str,
    cfg: IngestConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Document]:
    text = _strip_boilerplate(text, cfg.compiled_boilerplate())
    text = normalize_whitespace(text)
    if len(text) < cfg.min_chars:
        return None
    full_meta: Dict[str, Any] = {"source": source, "title": title}
    full_meta.update(cfg.extra_meta)
    if meta:
        full_meta.update(meta)
    return Document(
        doc_id=compute_doc_id(text, source),
        source=source,
        title=title or _first_heading(text),
        text=text,
        meta=full_meta,
    )


def _qa_markdown(question: str, answer: str) -> str:
    """Render a FAQ pair so the question and its answer cannot be separated."""
    q = normalize_whitespace(question.strip()).replace("\n", " ")
    a = normalize_whitespace(answer.strip())
    return "## %s\n\n%s" % (q, a)


def _load_csv(path: str, cfg: IngestConfig) -> List[Document]:
    raw = read_text_with_fallback(path)
    reader = csv.DictReader(io.StringIO(raw))
    docs: List[Document] = []
    for i, row in enumerate(reader):
        question = _pick(row, cfg.question_fields)
        answer = _pick(row, cfg.answer_fields)
        if not answer:
            continue
        if not question:
            question = "Entry %d" % (i + 1)
        extra = {
            k: v
            for k, v in row.items()
            if k
            and str(k).casefold() not in {n.casefold() for n in list(cfg.question_fields) + list(cfg.answer_fields)}
            and str(v or "").strip()
        }
        meta: Dict[str, Any] = {"kind": "faq_pair", "row": i + 1}
        meta.update(extra)
        doc = _make_doc(
            _qa_markdown(question, answer),
            "%s#row%d" % (path, i + 1),
            question.strip(),
            cfg,
            meta,
        )
        if doc:
            docs.append(doc)
    return docs


def _load_json(path: str, cfg: IngestConfig) -> List[Document]:
    raw = read_text_with_fallback(path)
    data = json.loads(raw)
    if isinstance(data, dict):
        for key in ("faqs", "items", "documents", "data", "entries"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("unsupported JSON shape in %s" % path)

    docs: List[Document] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = _pick(item, cfg.question_fields)
        answer = _pick(item, cfg.answer_fields)
        if answer is None:
            continue
        known = {n.casefold() for n in list(cfg.question_fields) + list(cfg.answer_fields)}
        meta: Dict[str, Any] = {"kind": "faq_pair", "row": i + 1}
        meta.update({k: v for k, v in item.items() if str(k).casefold() not in known})
        if question:
            text = _qa_markdown(question, answer)
            title = question.strip()
        else:
            text = str(answer)
            title = _first_heading(text)
        doc = _make_doc(text, "%s#item%d" % (path, i + 1), title, cfg, meta)
        if doc:
            docs.append(doc)
    return docs


def load_file(path: str, cfg: Optional[IngestConfig] = None) -> List[Document]:
    """Load one file into zero or more documents.

    Tabular sources (``.csv``, ``.json``) yield one document per FAQ pair;
    prose sources yield exactly one document.

    The path is normalised before it becomes the document source. Without that,
    ``docs/faq.md`` and ``docs/../docs/faq.md`` hash to two different documents
    and one re-ingest silently doubles the index.
    """
    cfg = cfg or IngestConfig()
    path = os.path.normpath(path)
    suffix = os.path.splitext(path)[1].casefold()
    if suffix == ".csv":
        return _load_csv(path, cfg)
    if suffix == ".json":
        return _load_json(path, cfg)

    raw = read_text_with_fallback(path)
    if suffix in (".html", ".htm"):
        text, title = html_to_text(raw, cfg.strip_html_tags)
        kind = "html"
    else:
        text, title = raw, ""
        kind = "markdown" if suffix in (".md", ".markdown") else "text"
    doc = _make_doc(
        text,
        path,
        title or _first_heading(text) or os.path.basename(path),
        cfg,
        {"kind": kind},
    )
    return [doc] if doc else []


def load_directory(
    root: str,
    cfg: Optional[IngestConfig] = None,
    *,
    recursive: bool = True,
    suffixes: Sequence[str] = SUPPORTED_SUFFIXES,
) -> List[Document]:
    """Load every supported file under ``root``, in sorted path order.

    Sorted order is deliberate: it makes chunk indices, and therefore the whole
    index, reproducible across machines and filesystems.
    """
    cfg = cfg or IngestConfig()
    root = os.path.normpath(root)
    allowed = {s.casefold() for s in suffixes}
    paths: List[str] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                if os.path.splitext(name)[1].casefold() in allowed:
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].casefold() in allowed:
                paths.append(full)
    docs: List[Document] = []
    for p in sorted(paths):
        docs.extend(load_file(p, cfg))
    return docs


def load_paths(paths: Iterable[str], cfg: Optional[IngestConfig] = None) -> List[Document]:
    """Load a mix of files and directories, de-duplicating by ``doc_id``.

    De-duplication here is what makes ``load_paths(x) == load_paths(x + x)``
    hold, which is the property that keeps a re-ingest from doubling the index.
    """
    cfg = cfg or IngestConfig()
    seen: Dict[str, Document] = {}
    for path in (os.path.normpath(p) for p in paths):
        if os.path.isdir(path):
            found = load_directory(path, cfg)
        elif os.path.isfile(path):
            found = load_file(path, cfg)
        else:
            raise FileNotFoundError(path)
        for doc in found:
            seen.setdefault(doc.doc_id, doc)
    return list(seen.values())
