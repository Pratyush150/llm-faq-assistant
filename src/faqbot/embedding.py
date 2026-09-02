"""Embedders. The default one has no dependencies and no network.

:class:`HashingEmbedder` is the reason this repository runs end to end with
zero API keys: it is a deterministic feature-hashing embedder built out of
character n-grams and word unigrams. It is not a neural model and does not
pretend to be one. What it *is*:

* deterministic — same text, same vector, on any machine, forever;
* offline and free — no key, no rate limit, no per-token cost;
* character-level — so ``NW-FILT-02`` and ``NW-FILT-O2`` are near neighbours,
  which is exactly the robustness a support corpus needs;
* L2-normalised — so dot product *is* cosine similarity.

What it is not: semantically aware. It will not know that "runtime" and
"battery life" mean the same thing. That is precisely why the retriever is
hybrid and why a real neural embedder is a one-line swap
(:class:`SentenceTransformerEmbedder`). The offline default keeps the tests
honest; the plugin keeps the system useful.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .textutil import ngrams, tokenize

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "GenericHTTPEmbedder",
    "register_embedder",
    "get_embedder",
    "available_embedders",
    "embedder_capabilities",
    "cosine",
    "l2_norm",
]


def l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Falls back to a full computation if inputs are not unit.

    Vectors produced by this module are already unit length, so this is a dot
    product in the common case.
    """
    if len(a) != len(b):
        raise ValueError("dimension mismatch: %d vs %d" % (len(a), len(b)))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    denom = math.sqrt(na) * math.sqrt(nb)
    if abs(denom - 1.0) < 1e-9:
        return dot
    return dot / denom


class Embedder(ABC):
    """Text to fixed-dimension vector.

    Implementations must return **L2-normalised** vectors so that the vector
    store can treat dot product as cosine similarity and skip a per-query
    normalisation pass.
    """

    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts, preserving order."""

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @classmethod
    def is_available(cls) -> bool:
        """Whether this embedder can actually run in this environment."""
        return True

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "dim": self.dim, "available": self.is_available()}

    def __repr__(self) -> str:
        return "%s(dim=%d)" % (type(self).__name__, self.dim)


class HashingEmbedder(Embedder):
    """Deterministic feature-hashing embedder. The offline default.

    Features are word unigrams (weighted higher, they carry topic) plus
    character n-grams (weighted lower, they carry robustness to typos and
    part-number variants). Each feature is hashed with BLAKE2b into a bucket
    and a sign, its weight is added with sublinear term-frequency damping
    (``1 + log(tf)``), and the vector is L2-normalised at the end.

    Signed hashing matters: with a single-sign scheme, hash collisions always
    add, so unrelated documents drift towards each other as the corpus grows.
    With signs, collisions cancel in expectation.

    Args:
        dim: Output dimension. 256 is enough for a few thousand chunks.
        ngram_range: Inclusive character n-gram sizes.
        word_weight: Weight applied to word-unigram features.
        char_weight: Weight applied to character n-gram features.
        seed: Changes the hash bucketing. Changing it invalidates any stored
            index, so it is recorded in the persisted index header.
    """

    name = "hashing"

    def __init__(
        self,
        dim: int = 256,
        ngram_range: Tuple[int, int] = (3, 5),
        *,
        word_weight: float = 1.0,
        char_weight: float = 0.55,
        seed: int = 0,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        lo, hi = ngram_range
        if lo < 1 or hi < lo:
            raise ValueError("invalid ngram_range %r" % (ngram_range,))
        self._dim = dim
        self.ngram_range = (lo, hi)
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.seed = seed
        self._prefix = struct.pack("<I", seed & 0xFFFFFFFF)

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, feature: str) -> Tuple[int, float]:
        digest = hashlib.blake2b(
            self._prefix + feature.encode("utf-8"), digest_size=8
        ).digest()
        value = int.from_bytes(digest, "big", signed=False)
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return (value % self._dim), sign

    def _features(self, text: str) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        for tok in tokenize(text):
            counts["w:" + tok] = counts.get("w:" + tok, 0.0) + self.word_weight
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for gram in ngrams(text, n):
                key = "c%d:%s" % (n, gram)
                counts[key] = counts.get(key, 0.0) + self.char_weight
        return counts

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for feature, weight in self._features(text or "").items():
                # Sublinear damping: a word repeated 30 times is not 30x the
                # evidence, and without damping long chunks dominate on their
                # own filler words.
                damped = (1.0 + math.log(weight)) if weight > 1.0 else weight
                idx, sign = self._bucket(feature)
                vec[idx] += sign * damped
            norm = l2_norm(vec)
            if norm > 0.0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class SentenceTransformerEmbedder(Embedder):
    """Local neural embeddings via ``sentence-transformers``. Guarded import.

    Swap this in when the corpus contains real paraphrase — customers asking
    "how long does it run" about a page that says "battery endurance". The
    hashing embedder cannot bridge that gap; a trained model can.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dim_cache: Optional[int] = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            return False

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "pip install sentence-transformers, or use the offline "
                    "'hashing' embedder."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim_cache is None:
            self._dim_cache = int(self._load().get_sentence_embedding_dimension())
        return self._dim_cache

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - optional dep
        model = self._load()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [[float(v) for v in row] for row in vectors]


class OpenAIEmbedder(Embedder):
    """Hosted embeddings. Guarded, and never invoked by the test suite.

    Requires ``OPENAI_API_KEY``. Construction is cheap and side-effect free;
    the client is only built on the first :meth:`embed` call, so importing this
    module in an environment with no key and no package is safe.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        *,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.model = model
        self._dim = dim
        self.api_key_env = api_key_env
        self._client: Any = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import importlib.util

            has_pkg = importlib.util.find_spec("openai") is not None
        except Exception:
            return False
        return has_pkg and bool(os.environ.get("OPENAI_API_KEY"))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - network
        if not os.environ.get(self.api_key_env):
            raise RuntimeError("%s is not set" % self.api_key_env)
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("pip install openai to use OpenAIEmbedder") from exc
            self._client = OpenAI()
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        out: List[List[float]] = []
        for item in resp.data:
            vec = [float(v) for v in item.embedding]
            norm = l2_norm(vec)
            out.append([v / norm for v in vec] if norm else vec)
        return out


class GenericHTTPEmbedder(Embedder):
    """Any JSON embedding endpoint. Guarded, and never invoked by the tests.

    For self-hosted servers (text-embeddings-inference, Ollama, vLLM). Posts
    ``{"input": [...]}`` and reads the vectors out with ``response_path``, a
    dotted path such as ``"data.*.embedding"``.
    """

    name = "http"

    def __init__(
        self,
        url: str,
        dim: int,
        *,
        headers: Optional[Dict[str, str]] = None,
        response_path: str = "data.*.embedding",
        timeout: float = 30.0,
        input_key: str = "input",
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.url = url
        self._dim = dim
        self.headers = dict(headers or {})
        self.response_path = response_path
        self.timeout = timeout
        self.input_key = input_key
        self.extra_payload = dict(extra_payload or {})

    @classmethod
    def is_available(cls) -> bool:
        # Constructible anywhere; whether the endpoint answers is a runtime
        # question, and probing it at import time would be a network call.
        return True

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _extract(payload: Any, path: str) -> List[List[float]]:
        node: Any = payload
        for part in path.split("."):
            if part == "*":
                if not isinstance(node, list):
                    raise ValueError("expected a list at '*' in %r" % path)
                rest = path.split(".", path.split(".").index("*") + 1)[-1]
                return [GenericHTTPEmbedder._extract(item, rest) for item in node]  # type: ignore[misc]
            node = node[part] if isinstance(node, dict) else node[int(part)]
        return node  # type: ignore[return-value]

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - network
        import json
        import urllib.request

        payload = dict(self.extra_payload)
        payload[self.input_key] = list(texts)
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        vectors = self._extract(body, self.response_path)
        out: List[List[float]] = []
        for vec in vectors:
            fvec = [float(v) for v in vec]
            norm = l2_norm(fvec)
            out.append([v / norm for v in fvec] if norm else fvec)
        return out


EMBEDDER_REGISTRY: Dict[str, Callable[..., Embedder]] = {
    "hashing": HashingEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
    "http": GenericHTTPEmbedder,
}


def register_embedder(name: str, factory: Callable[..., Embedder]) -> None:
    """Register a custom embedder factory under ``name``."""
    EMBEDDER_REGISTRY[name] = factory


def get_embedder(name: str = "hashing", **kwargs: Any) -> Embedder:
    """Build an embedder by registry name.

    Raises:
        KeyError: if ``name`` is unknown, listing the registered names.
    """
    try:
        factory = EMBEDDER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            "unknown embedder %r; available: %s" % (name, ", ".join(sorted(EMBEDDER_REGISTRY)))
        ) from None
    return factory(**kwargs)


def available_embedders() -> List[str]:
    """Names that can actually run here, cheapest capability check only."""
    out: List[str] = []
    for name, factory in sorted(EMBEDDER_REGISTRY.items()):
        checker = getattr(factory, "is_available", None)
        try:
            if checker is None or checker():
                out.append(name)
        except Exception:
            continue
    return out


def embedder_capabilities() -> Dict[str, Dict[str, Any]]:
    """Full capability report: what is registered, what is usable, and why not."""
    report: Dict[str, Dict[str, Any]] = {}
    for name, factory in sorted(EMBEDDER_REGISTRY.items()):
        checker = getattr(factory, "is_available", None)
        try:
            usable = True if checker is None else bool(checker())
        except Exception:
            usable = False
        report[name] = {
            "available": usable,
            "offline": name == "hashing",
            "needs_key": name == "openai",
            "class": getattr(factory, "__name__", str(factory)),
        }
    return report
