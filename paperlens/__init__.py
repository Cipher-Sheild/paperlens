"""PaperLens - BERT-powered research paper summarizer."""
from .config import SummarizerConfig, EMBEDDING_MODELS, ABSTRACTIVE_MODELS, EMBEDDING_ALIASES
from .pipeline import run_pipeline, run_from_file, run_from_source, benchmark_methods, SummaryResult

__version__ = "1.1.1"
__all__ = ["SummarizerConfig", "EMBEDDING_MODELS", "ABSTRACTIVE_MODELS", "EMBEDDING_ALIASES",
           "run_pipeline", "run_from_file", "run_from_source", "benchmark_methods", "SummaryResult"]
