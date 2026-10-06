"""End-to-end orchestration."""
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from .config import SummarizerConfig
from .preprocess import split_sentences, clean_text
from .embedder import get_embedder
from .extractive import select_for_groups, importance_scores, coverage_score, center_embeddings
from .scoring import quality_array
from .structure import assign_roles, structured_summary
from .evaluate import compare_lengths, rouge_table, keyword_retention


@dataclass
class SummaryResult:
    config: SummarizerConfig
    title: str
    source: str
    intro: str
    conclusion: str
    reference: Optional[str]
    sentences: list
    section: list
    groups: list
    embeddings: np.ndarray            # centered (if enabled) - used for clustering / charts
    quality: np.ndarray
    selected: list
    roles: dict
    importance: np.ndarray
    extractive_summary: str
    abstractive_summary: Optional[str]
    structured: dict
    stats_df: pd.DataFrame
    rouge_df: Optional[pd.DataFrame]
    coverage: float
    keyword_retention: float
    elapsed: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def summaries(self) -> dict:
        d = {f"Extractive ({self.config.method})": self.extractive_summary}
        if self.abstractive_summary:
            d["Abstractive (rewrite)"] = self.abstractive_summary
        return d


def run_pipeline(intro: str, conclusion: str, config: Optional[SummarizerConfig] = None,
                 reference: Optional[str] = None, embedder=None, title: str = "", source: str = "pasted text"
                 ) -> SummaryResult:
    t0 = time.perf_counter()
    cfg = config or SummarizerConfig()
    warnings = []
    intro_s = split_sentences(intro)[:cfg.max_input_sentences]
    concl_s = split_sentences(conclusion)[:cfg.max_input_sentences]
    sentences = intro_s + concl_s
    if len(sentences) < 3:
        raise ValueError("Not enough text: provide at least a few full sentences of Introduction/Conclusion.")
    section = ["Introduction"] * len(intro_s) + ["Conclusion"] * len(concl_s)
    groups = [list(range(len(intro_s))), list(range(len(intro_s), len(sentences)))]

    embedder = embedder or get_embedder(cfg.embedding_model)
    E_raw = embedder.encode(sentences)
    E = center_embeddings(E_raw) if cfg.center_embeddings else E_raw
    Q = quality_array(sentences)

    selected = select_for_groups(E, groups, cfg, sentences, Q=Q)
    extractive = " ".join(sentences[i] for i in selected)
    roles = assign_roles(sentences, section, selected, embedder, E_raw)

    abstractive = None
    if cfg.abstractive:
        try:
            from .abstractive import summarize_abstractive
            src = [sentences[i] for i in selected] if cfg.abstractive_input == "extractive" else sentences
            abstractive = summarize_abstractive(src, cfg.abstractive_model)
        except Exception as e:
            warnings.append(f"Abstractive step failed: {str(e)[:150]}")

    original = " ".join(sentences)
    summaries = {f"Extractive ({cfg.method})": extractive}
    if abstractive:
        summaries["Abstractive (rewrite)"] = abstractive

    return SummaryResult(
        config=cfg, title=title, source=source, intro=clean_text(intro), conclusion=clean_text(conclusion),
        reference=reference or None, sentences=sentences, section=section, groups=groups, embeddings=E,
        quality=Q, selected=selected, roles=roles, importance=importance_scores(E),
        extractive_summary=extractive, abstractive_summary=abstractive,
        structured=structured_summary(sentences, roles, selected),
        stats_df=compare_lengths(original, summaries), rouge_df=rouge_table(reference, summaries),
        coverage=coverage_score(E, selected), keyword_retention=keyword_retention(original, extractive),
        elapsed=time.perf_counter() - t0, warnings=warnings)


def run_from_file(path: str, config: Optional[SummarizerConfig] = None, reference: Optional[str] = None,
                  embedder=None, source: str = "") -> SummaryResult:
    from .pdf_utils import load_sections
    sec = load_sections(path)
    res = run_pipeline(sec.introduction, sec.conclusion, config, reference or sec.abstract or None, embedder,
                       title=sec.title, source=source or str(path))
    res.warnings = sec.warnings + res.warnings
    return res


def run_from_source(source: str, config: Optional[SummarizerConfig] = None, reference: Optional[str] = None,
                    embedder=None, allow_private: bool = False) -> SummaryResult:
    """`source` = local path, URL, or arXiv id."""
    import os
    if os.path.exists(source):
        return run_from_file(source, config, reference, embedder, source=source)
    from .fetch import download_pdf
    path = download_pdf(source, allow_private=allow_private)
    return run_from_file(path, config, reference, embedder, source=source)


def benchmark_methods(result: SummaryResult) -> pd.DataFrame:
    """Compare K-Means vs TextRank on the SAME embeddings (fast: no re-encoding)."""
    rows = {}
    for m in ("kmeans", "textrank"):
        chosen = select_for_groups(result.embeddings, result.groups, result.config, result.sentences,
                                   method=m, Q=result.quality)
        text = " ".join(result.sentences[i] for i in chosen)
        row = {"Sentences": len(chosen), "Words": len(text.split()),
               "Coverage": round(coverage_score(result.embeddings, chosen), 3)}
        rt = rouge_table(result.reference, {m: text}) if result.reference else None
        if rt is not None:
            row.update(rt.loc[m].to_dict())
        rows[m] = row
    return pd.DataFrame(rows).T
