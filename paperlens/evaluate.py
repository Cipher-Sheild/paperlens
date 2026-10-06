"""Length statistics, ROUGE, keyword retention."""
import re
from collections import Counter
import numpy as np
import pandas as pd

from .preprocess import STOPWORDS


def length_stats(text: str) -> dict:
    words = text.split()
    return {"Characters": len(text), "Words": len(words),
            "Sentences": max(1, len(re.findall(r"[.!?](?:\s|$)", text))),
            "Avg word length": round(float(np.mean([len(w) for w in words])), 2) if words else 0.0}


def compare_lengths(original: str, summaries: dict) -> pd.DataFrame:
    rows = {"Original": length_stats(original)}
    rows.update({name: length_stats(t) for name, t in summaries.items()})
    df = pd.DataFrame(rows).T
    ow = max(1, df.loc["Original", "Words"])
    df["Compression (%)"] = (df["Words"] / ow * 100).round(1)
    df["Reduction (%)"] = (100 - df["Compression (%)"]).round(1)
    return df


def rouge_table(reference: str, summaries: dict):
    """ROUGE F1 (in %) of each summary against a reference (e.g. the paper's own abstract)."""
    if not reference or not reference.strip():
        return None
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = {}
    for name, text in summaries.items():
        s = sc.score(reference, text)
        rows[name] = {"ROUGE-1": round(s["rouge1"].fmeasure * 100, 1),
                      "ROUGE-2": round(s["rouge2"].fmeasure * 100, 1),
                      "ROUGE-L": round(s["rougeL"].fmeasure * 100, 1)}
    return pd.DataFrame(rows).T


def top_keywords(text: str, n: int = 10) -> list:
    ws = [w for w in re.findall(r"[a-zA-Z\-]{3,}", text.lower()) if w not in STOPWORDS]
    return Counter(ws).most_common(n)


def keyword_retention(original: str, summary: str, n: int = 15) -> float:
    kws = [w for w, _ in top_keywords(original, n)]
    if not kws:
        return 0.0
    sw = set(re.findall(r"[a-zA-Z\-]{3,}", summary.lower()))
    return sum(k in sw for k in kws) / len(kws)
