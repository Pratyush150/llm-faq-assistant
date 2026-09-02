"""Chunking strategies, and why the default is not fixed-size.

The single most common reason a FAQ bot gives wrong answers is chunking, not
the model. Consider a support page::

    ## How long does the AR-1 battery last?

    Up to 90 minutes in Eco mode and about 45 minutes in Turbo mode. Runtime
    drops on deep carpet because the brush motor draws more current.

Split that at a fixed 40 tokens and you get two chunks. Chunk A is the heading
plus half a sentence. Chunk B is "...drops on deep carpet because the brush
motor draws more current." — a bare pronoun-free fragment with no subject.

Now the failure cascades:

* Embed chunk A and it looks like a *question*, not an answer. Questions
  embed close to other questions, so it will be retrieved for every battery
  query and will contain no numbers.
* Chunk B contains the real fact but has lost the words "battery" and
  "runtime", so neither the vector search nor BM25 will ever retrieve it for
  "how long does the battery last".
* The generator is then handed a heading and a fragment, and either says "I
  don't know" (best case) or invents "about 2 hours" (normal case).

The fix is structure. :class:`StructureAwareChunker` splits on the document's
own headings, keeps the heading path as breadcrumb metadata on every chunk, and
treats a question heading plus its answer body as an **atomic block** that is
never split unless it alone exceeds the size budget — and even then the split
pieces each keep the heading. The other two strategies are provided so the
evaluation harness can show, on your own corpus, how much worse they are.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .textutil import split_sentences, tokenize
from .types import Chunk, Document

__all__ = [
    "Chunker",
    "FixedTokenChunker",
    "SentenceChunker",
    "StructureAwareChunker",
    "get_chunker",
    "CHUNKER_REGISTRY",
    "chunk_documents",
    "Section",
    "parse_sections",
]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_QA_PREFIX_RE = re.compile(r"^\s*(q|question)\s*[:.)-]\s*(.+)$", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _chunk_id(doc_id: str, index: int, text: str) -> str:
    """Deterministic chunk id: same document + same position + same text."""
    h = hashlib.sha1()
    h.update(doc_id.encode("utf-8"))
    h.update(b"\x00%d\x00" % index)
    h.update(text.encode("utf-8"))
    return "%s-%s" % (doc_id[:8], h.hexdigest()[:10])


@dataclass
class Section:
    """A heading and the body directly beneath it."""

    breadcrumb: Tuple[str, ...]
    heading: str
    level: int
    body: str
    start_char: int
    end_char: int

    @property
    def is_question(self) -> bool:
        """True when this section's heading reads like a FAQ question.

        Question sections are the atomic units: splitting one is the bug this
        module exists to prevent.
        """
        h = self.heading.strip()
        if h.endswith("?"):
            return True
        if _QA_PREFIX_RE.match(h):
            return True
        return bool(
            re.match(
                r"^(how|what|why|when|where|who|which|can|do|does|is|are|will|should|may)\b",
                h,
                re.IGNORECASE,
            )
        )

    def render(self, include_heading: bool = True) -> str:
        if include_heading and self.heading:
            return ("%s\n\n%s" % (self.heading, self.body)).strip()
        return self.body.strip()


def parse_sections(text: str) -> List[Section]:
    """Split markdown into heading-scoped sections, tracking the heading path.

    Fenced code blocks are respected, so a ``#`` comment inside a shell example
    does not become a heading and shred the surrounding section.
    """
    lines = text.split("\n")
    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    sections: List[Section] = []
    path: List[Tuple[int, str]] = []
    cur_heading = ""
    cur_level = 0
    cur_lines: List[str] = []
    cur_start = 0
    in_fence = False

    def flush(end: int) -> None:
        body = "\n".join(cur_lines).strip()
        if not body and not cur_heading:
            return
        sections.append(
            Section(
                breadcrumb=tuple(h for _, h in path),
                heading=cur_heading,
                level=cur_level,
                body=body,
                start_char=cur_start,
                end_char=end,
            )
        )

    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush(offsets[i])
            level = len(m.group(1))
            heading = m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, heading))
            cur_heading = heading
            cur_level = level
            cur_lines = []
            cur_start = offsets[i]
        else:
            cur_lines.append(line)
    flush(len(text))
    return sections


def _token_count(text: str) -> int:
    """Word-ish token count.

    Deliberately not a BPE count. It is stable, dependency-free and within
    ~25% of a subword tokeniser for English prose, which is accurate enough to
    budget chunk sizes and does not tie the index to one vendor's vocabulary.
    """
    return len(tokenize(text))


class Chunker(ABC):
    """Turns a :class:`~faqbot.types.Document` into retrievable chunks."""

    name: str = "base"

    @abstractmethod
    def split(self, doc: Document) -> List[Chunk]:
        """Return the chunks for one document, in document order."""

    def __call__(self, doc: Document) -> List[Chunk]:
        return self.split(doc)

    def _make(
        self,
        doc: Document,
        index: int,
        text: str,
        *,
        breadcrumb: Sequence[str] = (),
        start: int = 0,
        end: int = 0,
        extra: Optional[Dict[str, object]] = None,
    ) -> Chunk:
        meta: Dict[str, object] = dict(doc.meta)
        meta.setdefault("source", doc.source)
        meta.setdefault("title", doc.title)
        meta["chunker"] = self.name
        if extra:
            meta.update(extra)
        return Chunk(
            chunk_id=_chunk_id(doc.doc_id, index, text),
            doc_id=doc.doc_id,
            text=text,
            index=index,
            breadcrumb=tuple(breadcrumb),
            start_char=start,
            end_char=end or (start + len(text)),
            meta=meta,
        )


class FixedTokenChunker(Chunker):
    """Fixed-width sliding window over tokens.

    Included as a baseline and a warning. It is fast, trivially parallel, and
    it will cut a question away from its answer roughly every ``max_tokens``
    tokens. Use it to measure how much worse it is than the structure-aware
    chunker on your corpus, not to serve traffic.
    """

    name = "fixed"

    def __init__(self, max_tokens: int = 120, overlap: int = 20) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= overlap < max_tokens:
            raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")
        self.max_tokens = max_tokens
        self.overlap = overlap

    def split(self, doc: Document) -> List[Chunk]:
        words = doc.text.split()
        if not words:
            return []
        step = self.max_tokens - self.overlap
        chunks: List[Chunk] = []
        index = 0
        start = 0
        while start < len(words):
            window = words[start : start + self.max_tokens]
            text = " ".join(window)
            chunks.append(self._make(doc, index, text, breadcrumb=(doc.title,) if doc.title else ()))
            index += 1
            if start + self.max_tokens >= len(words):
                break
            start += step
        return chunks


class SentenceChunker(Chunker):
    """Pack whole sentences up to a token budget, with sentence-level overlap.

    Better than fixed-width because it never cuts mid-sentence, so no chunk is
    a dangling fragment. It still does not know that a heading belongs to the
    text under it, so it will happily end a chunk immediately after a question
    heading.
    """

    name = "sentence"

    def __init__(self, max_tokens: int = 120, overlap_sentences: int = 1) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def split(self, doc: Document) -> List[Chunk]:
        sentences = split_sentences(doc.text)
        if not sentences:
            return []
        chunks: List[Chunk] = []
        buf: List[str] = []
        budget = 0
        index = 0
        for sent in sentences:
            n = _token_count(sent)
            if buf and budget + n > self.max_tokens:
                chunks.append(self._make(doc, index, "\n".join(buf), breadcrumb=(doc.title,) if doc.title else ()))
                index += 1
                buf = buf[-self.overlap_sentences :] if self.overlap_sentences else []
                budget = sum(_token_count(s) for s in buf)
            buf.append(sent)
            budget += n
        if buf:
            chunks.append(self._make(doc, index, "\n".join(buf), breadcrumb=(doc.title,) if doc.title else ()))
        return chunks


class StructureAwareChunker(Chunker):
    """Heading-scoped chunking. This is the default.

    Rules, in order:

    1. Parse the document into sections using its own heading hierarchy.
    2. A section whose heading is a question (``ends with "?"``, ``Q:`` prefix,
       or a leading interrogative word) is **atomic**: heading and body stay in
       one chunk whenever they fit within ``max_tokens * hard_limit_factor``.
       This is the rule that stops a Q being separated from its A.
    3. A non-question section that fits is emitted whole; small adjacent
       sections under the same parent are merged up to the budget so the index
       is not full of two-line chunks that match nothing.
    4. A section that is genuinely too long is split on sentence boundaries,
       and **every** piece is re-prefixed with its heading and keeps the full
       breadcrumb, so no piece is context-free.
    """

    name = "structure"

    def __init__(
        self,
        max_tokens: int = 180,
        overlap_sentences: int = 1,
        *,
        min_tokens: int = 24,
        hard_limit_factor: float = 2.0,
        prefix_breadcrumb: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if hard_limit_factor < 1.0:
            raise ValueError("hard_limit_factor must be >= 1.0")
        self.max_tokens = max_tokens
        self.overlap_sentences = max(0, overlap_sentences)
        self.min_tokens = max(0, min_tokens)
        self.hard_limit_factor = hard_limit_factor
        self.prefix_breadcrumb = prefix_breadcrumb

    def _render(self, section: Section, body: str) -> str:
        """Breadcrumb, heading and body as three blank-line separated blocks.

        The blank lines are load-bearing. Downstream, sentence splitting treats
        a blank line as a hard boundary, so keeping them separate is what stops
        the answerer from quoting "[Battery] How long does the battery last?"
        back at the user as if it were the answer.
        """
        parts: List[str] = []
        if self.prefix_breadcrumb and len(section.breadcrumb) > 1:
            parts.append("[%s]" % " > ".join(section.breadcrumb[:-1]))
        if section.heading:
            parts.append(section.heading)
        parts.append(body)
        return "\n\n".join(p for p in parts if p).strip()

    def split(self, doc: Document) -> List[Chunk]:
        sections = parse_sections(doc.text)
        if not sections:
            return SentenceChunker(self.max_tokens, self.overlap_sentences).split(doc)

        hard_limit = int(self.max_tokens * self.hard_limit_factor)
        chunks: List[Chunk] = []
        index = 0
        pending: Optional[Section] = None
        pending_text = ""

        def emit(section: Section, text: str, part: int = 0, parts: int = 1) -> None:
            nonlocal index
            chunks.append(
                self._make(
                    doc,
                    index,
                    text,
                    breadcrumb=section.breadcrumb or ((doc.title,) if doc.title else ()),
                    start=section.start_char,
                    end=section.end_char,
                    extra={
                        "heading": section.heading,
                        "heading_level": section.level,
                        "is_question": section.is_question,
                        "part": part,
                        "parts": parts,
                        "atomic": parts == 1,
                    },
                )
            )
            index += 1

        def flush_pending() -> None:
            nonlocal pending, pending_text
            if pending is not None and pending_text.strip():
                emit(pending, pending_text.strip())
            pending = None
            pending_text = ""

        for section in sections:
            body = section.body.strip()
            if not body:
                # A heading with nothing under it is a container, not content.
                # Indexing it would create a chunk that matches on the heading
                # words and carries no answer. Its text survives in the
                # breadcrumb of every child section.
                continue
            whole = self._render(section, body)
            size = _token_count(whole)

            if size > hard_limit and not (section.is_question and size <= hard_limit * 1.5):
                flush_pending()
                for part_i, piece in enumerate(self._split_body(body)):
                    emit(section, self._render(section, piece), part_i, 0)
                # Backfill the real part count now that the section is done.
                total = sum(1 for c in chunks if c.meta.get("parts") == 0)
                for c in chunks:
                    if c.meta.get("parts") == 0:
                        c.meta["parts"] = total
                        c.meta["atomic"] = False
                continue

            if section.is_question or size >= self.min_tokens:
                flush_pending()
                emit(section, whole)
                continue

            # Small non-question section: merge forward until it is worth indexing.
            if pending is None:
                pending = section
                pending_text = whole
            else:
                merged = pending_text + "\n\n" + whole
                if _token_count(merged) > self.max_tokens:
                    flush_pending()
                    pending = section
                    pending_text = whole
                else:
                    pending_text = merged
                    pending = Section(
                        breadcrumb=pending.breadcrumb,
                        heading=pending.heading,
                        level=pending.level,
                        body=pending.body,
                        start_char=pending.start_char,
                        end_char=section.end_char,
                    )
        flush_pending()
        return chunks

    def _split_body(self, body: str) -> List[str]:
        sentences = split_sentences(body)
        if not sentences:
            return [body]
        pieces: List[str] = []
        buf: List[str] = []
        budget = 0
        for sent in sentences:
            n = _token_count(sent)
            if buf and budget + n > self.max_tokens:
                pieces.append("\n".join(buf))
                buf = buf[-self.overlap_sentences :] if self.overlap_sentences else []
                budget = sum(_token_count(s) for s in buf)
            buf.append(sent)
            budget += n
        if buf:
            pieces.append("\n".join(buf))
        return pieces


CHUNKER_REGISTRY: Dict[str, type] = {
    "fixed": FixedTokenChunker,
    "sentence": SentenceChunker,
    "structure": StructureAwareChunker,
}


def get_chunker(name: str, **kwargs: object) -> Chunker:
    """Look up a chunker by registry name.

    Raises:
        KeyError: if ``name`` is not registered, listing what is.
    """
    try:
        cls = CHUNKER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            "unknown chunker %r; available: %s" % (name, ", ".join(sorted(CHUNKER_REGISTRY)))
        ) from None
    return cls(**kwargs)  # type: ignore[arg-type]


def chunk_documents(docs: Iterable[Document], chunker: Chunker) -> List[Chunk]:
    """Chunk many documents, preserving document order."""
    out: List[Chunk] = []
    for doc in docs:
        out.extend(chunker.split(doc))
    return out
