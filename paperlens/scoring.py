"""Sentence quality: prefer informative, self-contained sentences over dangling or trivia sentences."""
import re
import numpy as np

CUES = ("this paper", "this study", "this review", "this work", "this article", "this research", "we propose",
        "we present", "we introduce", "we show", "we find", "we found", "we conclude", "our results", "our findings",
        "results show", "results indicate", "findings", "in conclusion", "to conclude", "in summary", "overall",
        "contribution", "aims to", "the aim of", "the purpose", "demonstrate", "reveal")
DANGLING = ("however", "thus", "therefore", "moreover", "furthermore", "additionally", "also", "nonetheless",
            "nevertheless", "consequently", "hence", "these", "those", "such", "it ", "they", "he ", "she ",
            "in addition", "for example", "for instance", "similarly", "meanwhile")


def quality_score(sentence: str) -> float:
    """Roughly in [-0.4, +0.15]: bonus for cue phrases, penalties for dangling starts, odd length, trivia."""
    t = sentence.lower().strip()
    q = 0.0
    has_cue = any(c in t for c in CUES)
    if has_cue:
        q += 0.15
    if t.startswith(DANGLING) and not has_cue:
        q -= 0.20
    n = len(sentence.split())
    if n < 8 or n > 45:
        q -= 0.15
    if re.search(r"[\[\]]", sentence):
        q -= 0.10
    if re.search(r"\b1[0-9]{3}\b", sentence) and not has_cue:      # historical trivia (e.g. "in 1956 ...")
        q -= 0.10
    return q


def quality_array(sentences) -> np.ndarray:
    return np.array([quality_score(s) for s in sentences], dtype=np.float32)
