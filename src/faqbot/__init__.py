"""faqbot — a retrieval-grounded FAQ assistant that measures itself.

Runs end to end with no API key and no network: a deterministic hashing
embedder and an extractive answerer are the defaults, and every hosted
provider is an optional, guarded plugin.

Quick start::

    from faqbot import FAQPipeline

    pipe = FAQPipeline()
    pipe.ingest(["data/corpus"])
    answer = pipe.ask("How long does the AR-1 battery last?")
    print(answer.render())
"""

from .answer import ExtractiveAnswerer, LLMAnswerer
from .chunking import FixedTokenChunker, SentenceChunker, StructureAwareChunker, get_chunker
from .embedding import Embedder, HashingEmbedder, get_embedder
from .eval import EvalReport, GoldSet, compare, load_goldset, run_eval
from .guardrails import GuardrailConfig, Guardrails
from .ingest import IngestConfig, load_paths
from .pipeline import ConversationMemory, FAQPipeline, PipelineConfig, QueryRewriter
from .rerank import FeatureReranker
from .store import BM25Index, HybridRetriever, VectorStore
from .types import Answer, Chunk, Citation, Document, RefusalReason, ScoredChunk

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Answer",
    "BM25Index",
    "Chunk",
    "Citation",
    "ConversationMemory",
    "Document",
    "Embedder",
    "EvalReport",
    "ExtractiveAnswerer",
    "FAQPipeline",
    "FeatureReranker",
    "FixedTokenChunker",
    "GoldSet",
    "GuardrailConfig",
    "Guardrails",
    "HashingEmbedder",
    "HybridRetriever",
    "IngestConfig",
    "LLMAnswerer",
    "PipelineConfig",
    "QueryRewriter",
    "RefusalReason",
    "ScoredChunk",
    "SentenceChunker",
    "StructureAwareChunker",
    "VectorStore",
    "compare",
    "get_chunker",
    "get_embedder",
    "load_goldset",
    "load_paths",
    "run_eval",
]
