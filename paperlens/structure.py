"""Label selected sentences: Research Problem, Approach, Major Findings, Conclusion.
Uses BERT similarity to prototype sentences (works for reviews too), plus a light keyword/section prior."""
import numpy as np

from .embedder import _unit

ROLES = ["Research Problem", "Approach", "Major Findings", "Conclusion"]

PROTOTYPES = {
    "Research Problem": [
        "Reading and finding the main contribution is difficult and time consuming.",
        "However, existing approaches are limited and fail to address this challenge.",
        "The rapid growth of information creates a strong need for better tools.",
        "Little is known about this issue, which motivates the present work.",
    ],
    "Approach": [
        "In this paper, we propose a new method based on a pre-trained model.",
        "We present a framework that combines several techniques to solve the task.",
        "This review examines and synthesizes existing studies on the topic.",
        "The study follows a systematic methodology to collect and analyse the data.",
    ],
    "Major Findings": [
        "The results show that the proposed method achieves better performance.",
        "Experiments demonstrate significant improvements over the baselines.",
        "The analysis reveals clear benefits as well as important challenges.",
        "We found that the approach is effective and produces consistent outcomes.",
    ],
    "Conclusion": [
        "In conclusion, this work shows the potential of the approach.",
        "Overall, we conclude that the method is effective, with some limitations.",
        "Future research should explore these directions further.",
        "These findings have important implications for practice and policy.",
    ],
}

KEYWORDS = {
    "Research Problem": ["problem", "challenge", "difficult", "however", "lack", "limited", "gap", "need"],
    "Approach": ["we propose", "we present", "we use", "method", "approach", "framework", "review", "study examines"],
    "Major Findings": ["results", "show", "demonstrate", "achieve", "outperform", "reveal", "found", "%"],
    "Conclusion": ["conclude", "overall", "future", "limitation", "implication", "in summary"],
}
_PRIOR = {"Introduction": {"Research Problem": 0.04, "Approach": 0.02},
          "Conclusion": {"Major Findings": 0.03, "Conclusion": 0.05}}


def assign_roles(sentences, sections, selected, embedder, E_raw) -> dict:
    """Return {sentence_index: role} for the selected sentences."""
    texts, owner = [], []
    for role, ps in PROTOTYPES.items():
        texts += ps
        owner += [role] * len(ps)
    P = embedder.encode(texts)
    mu = np.vstack([E_raw, P]).mean(axis=0, keepdims=True)
    Ec, Pc = _unit(E_raw - mu), _unit(P - mu)
    roles = {}
    for i in selected:
        sims = Ec[i] @ Pc.T
        scores = {r: float(max(s for s, o in zip(sims, owner) if o == r)) for r in ROLES}
        low = sentences[i].lower()
        for r in ROLES:
            scores[r] += 0.03 * sum(k in low for k in KEYWORDS[r])
            scores[r] += _PRIOR.get(sections[i], {}).get(r, 0.0)
        roles[i] = max(scores, key=scores.get)
    return roles


def structured_summary(sentences, roles: dict, selected) -> dict:
    out = {r: [] for r in ROLES}
    for i in selected:
        out[roles[i]].append(sentences[i])
    return {r: v for r, v in out.items() if v}
