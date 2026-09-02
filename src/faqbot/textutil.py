"""Tokenisation, sentence splitting and text normalisation.

One module, so that the tokeniser used by BM25 is provably the same tokeniser
used by the hashing embedder, the reranker and the grounding check. When those
drift apart you get a system where the keyword index and the vector index
disagree about what a word is, and nobody notices until a part number stops
matching.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Set, Tuple

__all__ = [
    "locate_span",
    "longest_common_phrase",
    "normalize_whitespace",
    "normalize_text",
    "tokenize",
    "token_set",
    "split_sentences",
    "ngrams",
    "jaccard",
    "coverage",
    "light_stem",
    "stem_tokens",
    "STOPWORDS",
]

# Small, deliberate stopword list. It is short on purpose: aggressive stopword
# removal deletes exactly the words that disambiguate FAQ questions ("can I
# *not* ...", "how *long* ..."). These are the ones that carry no signal at all.
STOPWORDS: Set[str] = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "am", "it", "its", "this", "that", "these",
    "those", "as", "at", "by", "from", "with", "i", "you", "we", "they", "he",
    "she", "do", "does", "did", "my", "your", "our", "their",
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")

# Sentence terminator followed by whitespace and something that starts a new
# sentence. The negative lookbehind list covers the abbreviations that actually
# show up in product documentation and otherwise chop a sentence in half.
_ABBREV = (
    "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "vs", "etc", "e.g",
    "i.e", "fig", "no", "approx", "min", "max", "hr", "hrs", "sec", "cf",
)
_SENT_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

# A column gap: two or more spaces between non-space text, i.e. a table row in
# a plain-text document rather than a hard-wrapped prose line.
_TABLE_ROW_RE = re.compile(r"\S {2,}\S")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, trim lines, cap blank-line runs at one.

    Markdown structure (headings, list markers, fenced blocks) is preserved;
    only horizontal noise and vertical padding are removed.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("​", "")
    # Runs of two or more spaces collapse to exactly two, not one. That gap is
    # the only surviving evidence that a plain-text line is a table row
    # ("Runtime               up to 90 minutes"), and sentence splitting needs
    # it: without it a 20-row spec sheet becomes one 300-word "sentence" and
    # the answerer quotes the entire table back at the user.
    lines = [
        re.sub(r"[ \t]{2,}", "  ", line.replace("\t", " ")).rstrip()
        for line in text.split("\n")
    ]
    out: List[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank == 1:
                out.append("")
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def normalize_text(text: str) -> str:
    """Unicode NFKC + case fold + whitespace collapse, for matching only.

    Never store the result as the document body: NFKC rewrites typographic
    quotes and dashes, which is fine for comparison and wrong for display.
    """
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def tokenize(text: str, *, drop_stopwords: bool = False) -> List[str]:
    """Lowercase word tokens, keeping identifier-ish glue characters.

    ``AR-1``, ``NW-FILT-02`` and ``v2.3`` survive as single tokens. That is the
    whole point: those are the strings a customer types when something is
    broken, and a tokeniser that shatters them into ``ar`` + ``1`` is the
    reason keyword search "mysteriously" fails on part numbers.
    """
    toks = _WORD_RE.findall(normalize_text(text))
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def token_set(text: str, *, drop_stopwords: bool = True) -> Set[str]:
    return set(tokenize(text, drop_stopwords=drop_stopwords))


def split_sentences(text: str) -> List[str]:
    """Split into sentences. Headings and list items are units of their own.

    Consecutive prose lines are joined before splitting. This matters more than
    it sounds: documentation is hard-wrapped at 80 columns, and splitting on
    newlines instead of on sentence boundaries yields fragments like "Runtime
    is up to 90 minutes in Eco mode and about 45 minutes in Turbo mode on" —
    a citation quote that stops mid-clause is worse than no quote at all.

    Returns stripped, non-empty sentences in document order.
    """
    out: List[str] = []
    para: List[str] = []

    def flush() -> None:
        if not para:
            return
        joined = " ".join(para)
        para.clear()
        for sent in _split_prose(joined):
            if sent:
                out.append(sent)

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if (
            line.startswith("#")
            or re.match(r"^([-*+]|\d+[.)])\s", line)
            or line.startswith("|")
            or _TABLE_ROW_RE.search(line)
        ):
            flush()
            out.append(line)
            continue
        para.append(line)
    flush()
    return out


def _split_prose(block: str) -> List[str]:
    """Sentence-split one joined paragraph, respecting common abbreviations."""
    parts = _SENT_END_RE.split(block)
    out: List[str] = []
    buf = ""
    for i, part in enumerate(parts):
        cand = (buf + " " + part).strip() if buf else part.strip()
        tail = cand.rstrip("\"')]").rstrip()
        words = tail.split()
        last = words[-1].rstrip(".").casefold() if words else ""
        if last in _ABBREV and i < len(parts) - 1:
            buf = cand
            continue
        buf = ""
        if cand:
            out.append(cand)
    if buf:
        out.append(buf)
    return out


def locate_span(haystack: str, needle: str, start: int = 0) -> Tuple[int, int]:
    """Find ``needle`` in ``haystack`` tolerating differences in whitespace.

    :func:`split_sentences` joins hard-wrapped lines, so a returned sentence
    usually does not appear verbatim in the source text — the source has a
    newline where the sentence has a space. Citations quote character offsets
    into the original chunk, and an offset that is merely nearby produces a
    highlight that starts mid-word. Falls back to ``(start, start)`` when the
    sentence genuinely is not there.
    """
    if not needle:
        return start, start
    idx = haystack.find(needle, start)
    if idx >= 0:
        return idx, idx + len(needle)
    pattern = r"\s+".join(re.escape(word) for word in needle.split())
    m = re.compile(pattern).search(haystack, start)
    if m:
        return m.start(), m.end()
    return start, start


def ngrams(text: str, n: int) -> List[str]:
    """Character n-grams over normalised text, with word-boundary padding."""
    padded = " " + normalize_text(text) + " "
    if len(padded) < n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def light_stem(token: str) -> str:
    """Strip the handful of English suffixes that break exact matching.

    Not a real stemmer, and deliberately so. Porter-style stemming conflates
    words a support corpus needs to keep apart (``charges``/``charging`` is
    fine; ``mopping``/``mop`` is fine; ``batteries``/``batter`` is not). This
    handles plurals and the ``-ing``/``-ed`` pair on long words and stops
    there, which is enough to make "error code" match "error codes" without
    inventing collisions.

    Applied to lexical *comparison* only. BM25 and the hashing embedder index
    raw tokens, so identifiers like ``NW-BAG-05`` are never mangled.
    """
    if len(token) <= 3 or any(ch.isdigit() for ch in token) or "-" in token:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("es") and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def stem_tokens(tokens: Iterable[str]) -> Set[str]:
    """Light-stemmed token set, for lexical comparison."""
    return {light_stem(t) for t in tokens}


def coverage(query_tokens: Sequence[str], text_tokens: Iterable[str]) -> float:
    """Fraction of the query's distinct content tokens present in the text.

    Asymmetric on purpose. A long chunk should not be penalised for containing
    words the question did not use; it should be rewarded for containing the
    words the question did use. Both sides are light-stemmed so that "error
    code" matches a section headed "Error codes".
    """
    q = stem_tokens(t for t in query_tokens if t not in STOPWORDS)
    if not q:
        return 0.0
    t = stem_tokens(text_tokens)
    return len(q & t) / float(len(q))


def longest_common_phrase(a: str, b: str, *, min_tokens: int = 2) -> Tuple[int, str]:
    """Longest shared contiguous token run between two strings.

    Used for the exact-phrase bonus in the reranker and for the answer
    grounding check, where "the sentence shares five consecutive tokens with a
    cited chunk" is much stronger evidence than bag-of-words overlap.
    """
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0, ""
    best_len = 0
    best_i = 0
    prev = [0] * (len(tb) + 1)
    for i in range(1, len(ta) + 1):
        cur = [0] * (len(tb) + 1)
        for j in range(1, len(tb) + 1):
            if ta[i - 1] == tb[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len = cur[j]
                    best_i = i
        prev = cur
    if best_len < min_tokens:
        return best_len, ""
    return best_len, " ".join(ta[best_i - best_len : best_i])
