"""Central configuration: model choices and summarizer settings."""
from dataclasses import dataclass

# Friendly label -> Hugging Face model id (used by the web UI)
EMBEDDING_MODELS = {
    "Sentence-BERT MiniLM - fast & accurate (recommended)": "sentence-transformers/all-MiniLM-L6-v2",
    "BERT base - the classic (bert-base-uncased)": "bert-base-uncased",
    "SciBERT - trained on scientific papers": "allenai/scibert_scivocab_uncased",
    "Baseline - word counts, no BERT (offline)": "hashing",
}

# Short aliases (used by the CLI)
EMBEDDING_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bert": "bert-base-uncased",
    "scibert": "allenai/scibert_scivocab_uncased",
    "hashing": "hashing",
}

ABSTRACTIVE_MODELS = {
    "DistilBART CNN (fast)": "sshleifer/distilbart-cnn-12-6",
    "BART large CNN (better, slower)": "facebook/bart-large-cnn",
}

METHODS = ("kmeans", "textrank")


@dataclass
class SummarizerConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"   # BERT-style encoder or "hashing"
    method: str = "kmeans"                 # "kmeans" or "textrank"
    target_words: int = 180                # desired summary length (approximate)
    conclusion_share: float = 0.4          # share of the summary reserved for the Conclusion
    use_quality: bool = True               # prefer self-contained, informative sentences
    center_embeddings: bool = True         # remove BERT's common direction -> sharper similarities
    max_input_sentences: int = 120         # safety cap per section
    abstractive: bool = False              # optional rewrite step
    abstractive_model: str = "sshleifer/distilbart-cnn-12-6"
    abstractive_input: str = "extractive"  # "extractive" (hybrid) or "full"

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not 40 <= self.target_words <= 800:
            raise ValueError("target_words must be between 40 and 800")
        if not 0.1 <= self.conclusion_share <= 0.7:
            raise ValueError("conclusion_share must be between 0.1 and 0.7")
