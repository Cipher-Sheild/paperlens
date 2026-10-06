"""Extractive sentence selection on top of sentence embeddings."""
import numpy as np
from sklearn.cluster import KMeans

from .embedder import _unit


def center_embeddings(E: np.ndarray) -> np.ndarray:
    """Subtract the mean vector and re-normalise. Raw BERT vectors share a big common direction, which makes
    every sentence look similar (cosine 0.6-1.0). Centering restores contrast."""
    if len(E) < 3:
        return _unit(E)
    return _unit(E - E.mean(axis=0, keepdims=True))


def sentences_needed(sentences: list, idx: list, budget_words: float) -> int:
    avg = max(1.0, float(np.mean([len(sentences[i].split()) for i in idx])))
    return int(max(1, min(len(idx), round(budget_words / avg))))


def select_kmeans(E, idx, k, Q=None, seed=42):
    """One sentence per cluster: the one that is both central to its cluster and high quality."""
    if k >= len(idx):
        return sorted(idx)
    X = E[idx]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    chosen = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1)
        closeness = 1.0 - (d - d.min()) / (np.ptp(d) + 1e-9)
        q = np.array([Q[idx[m]] for m in members]) if Q is not None else 0.0
        chosen.append(idx[members[int(np.argmax(0.5 * closeness + q))]])
    return sorted(set(chosen))


def select_textrank(E, idx, k, Q=None, damping=0.85, iters=100):
    """PageRank on the cosine-similarity graph, combined with sentence quality."""
    if k >= len(idx):
        return sorted(idx)
    X = _unit(E[idx])
    S = np.clip(X @ X.T, 0, None)
    np.fill_diagonal(S, 0)
    rs = S.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    T = S / rs
    n = len(idx)
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        new = (1 - damping) / n + damping * (T.T @ r)
        if np.abs(new - r).sum() < 1e-9:
            r = new
            break
        r = new
    score = 0.5 * (r / (r.max() + 1e-12))
    if Q is not None:
        score = score + np.array([Q[i] for i in idx])
    return sorted(idx[i] for i in np.argsort(-score)[:k])


def select(E, idx, k, method="kmeans", Q=None):
    return select_textrank(E, idx, k, Q) if method == "textrank" else select_kmeans(E, idx, k, Q)


def select_for_groups(E, groups, cfg, sentences, method=None, Q=None) -> list:
    """Pick sentences per section. The word budget is split so the Conclusion always gets its share."""
    method = method or cfg.method
    live = [g for g in groups if g]
    chosen = []
    for gi, idx in enumerate(groups):
        if not idx:
            continue
        if len(live) == 1:
            share = 1.0
        else:
            share = (1 - cfg.conclusion_share) if gi == 0 else cfg.conclusion_share
        k = sentences_needed(sentences, idx, cfg.target_words * share)
        chosen += select(E, idx, k, method, Q if cfg.use_quality else None)
    return sorted(set(chosen))


def importance_scores(E: np.ndarray) -> np.ndarray:
    """Centrality: average similarity of a sentence to all the other sentences."""
    n = len(E)
    if n < 2:
        return np.zeros(n)
    X = _unit(E)
    S = X @ X.T
    return (S.sum(axis=1) - 1.0) / (n - 1)


def coverage_score(E: np.ndarray, selected: list) -> float:
    """How well the selected sentences represent ALL sentences (mean best-match cosine)."""
    X = _unit(E)
    return float((X @ X[selected].T).max(axis=1).mean())
