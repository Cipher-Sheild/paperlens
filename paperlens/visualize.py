"""One-figure dashboard."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .evaluate import top_keywords


def dashboard(res):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("PaperLens - Summary Dashboard", fontsize=18, fontweight="bold")
    df = res.stats_df
    names = list(df.index)
    palette = ["#4C72B0", "#55A868", "#C44E52"][:len(names)]
    sel = set(res.selected)
    n = len(res.sentences)

    ax = axes[0, 0]
    b = ax.bar(names, df["Words"], color=palette); ax.bar_label(b, fontweight="bold")
    ax.set_title("1. Word count"); ax.set_ylabel("Words"); ax.tick_params(axis="x", labelsize=8)

    ax = axes[0, 1]
    b = ax.bar(names, df["Compression (%)"], color=palette); ax.bar_label(b, fmt="%.1f%%", fontweight="bold")
    ax.set_ylim(0, 118); ax.set_title("2. Length as % of original"); ax.tick_params(axis="x", labelsize=8)

    ax = axes[0, 2]
    ax.bar(range(n), res.importance, color=["#2ca02c" if i in sel else "#c7c7c7" for i in range(n)],
           edgecolor=["#1f77b4" if s == "Introduction" else "#ff7f0e" for s in res.section], linewidth=2)
    first_concl = res.section.index("Conclusion") if "Conclusion" in res.section else None
    if first_concl:
        ax.axvline(first_concl - 0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("Sentence (blue border=Intro, orange=Conclusion)")
    ax.set_ylabel("Centrality (avg. similarity to others)"); ax.set_title("3. Sentence centrality (green=selected)")

    ax = axes[1, 0]
    P = PCA(n_components=2, random_state=42).fit_transform(res.embeddings)
    ax.scatter(P[:, 0], P[:, 1], c=["#1f77b4" if s == "Introduction" else "#ff7f0e" for s in res.section],
               s=70, alpha=0.7)
    sl = list(sel)
    ax.scatter(P[sl, 0], P[sl, 1], marker="*", s=380, c="gold", edgecolors="k", zorder=3)
    for i, (x, y) in enumerate(P):
        ax.annotate(str(i), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("4. PCA of sentence embeddings (star=selected)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax = axes[1, 1]
    S = res.embeddings @ res.embeddings.T
    im = ax.imshow(S, cmap="viridis"); fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("5. Sentence similarity heatmap"); ax.set_xlabel("Sentence"); ax.set_ylabel("Sentence")

    ax = axes[1, 2]
    original = " ".join(res.sentences)
    tw = top_keywords(original, 10)[::-1]
    sw = set(re.findall(r"[a-zA-Z\-]{3,}", res.extractive_summary.lower()))
    ax.barh([w for w, _ in tw], [c for _, c in tw], color=["#2ca02c" if w in sw else "#d62728" for w, _ in tw])
    ax.set_title(f"6. Top keywords (green=kept, {res.keyword_retention:.0%} of top-15 retained)")
    ax.set_xlabel("Frequency")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
