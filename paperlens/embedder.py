"""Sentence embedders. BertEmbedder wraps any BERT-style Hugging Face encoder with mean pooling."""
import hashlib
import re
import numpy as np

from .preprocess import STOPWORDS


def _unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


class HashingEmbedder:
    """Tiny bag-of-words baseline (no neural network). Used for tests, offline demos and
    as a baseline to show how much BERT improves over word counts."""
    name = "hashing"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, sentences):
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            for w in re.findall(r"[a-z]{3,}", s.lower()):
                if w in STOPWORDS:
                    continue
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _unit(out)


class BertEmbedder:
    """Mean-pooled, L2-normalised sentence embeddings from a pre-trained BERT-style encoder."""

    def __init__(self, model_name="bert-base-uncased", max_length=128, batch_size=16, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.name = model_name
        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode(self, sentences):
        torch = self.torch
        chunks = []
        with torch.no_grad():
            for i in range(0, len(sentences), self.batch_size):
                batch = list(sentences[i:i + self.batch_size])
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                chunks.append(pooled.cpu().numpy())
        return _unit(np.vstack(chunks))


_CACHE = {}


def get_embedder(model_name: str):
    """Return a cached embedder so a model is only loaded once per process."""
    if model_name not in _CACHE:
        _CACHE[model_name] = HashingEmbedder() if model_name == "hashing" else BertEmbedder(model_name)
    return _CACHE[model_name]
