"""Markdown / JSON export."""
import json


def build_markdown(res) -> str:
    L = ["# PaperLens Summary Report", ""]
    if res.title:
        L += [f"**Paper:** {res.title}", ""]
    L += [f"- Source: {res.source}",
          f"- Embedding model: `{res.config.embedding_model}`",
          f"- Selection: `{res.config.method}`, target {res.config.target_words} words",
          f"- Coverage: {res.coverage:.3f}  |  Keyword retention: {res.keyword_retention:.0%}", ""]
    if res.warnings:
        L += ["## Notes"] + [f"- {w}" for w in res.warnings] + [""]
    L += ["## Summary (BERT extractive)", "", res.extractive_summary, ""]
    if res.abstractive_summary:
        L += ["## Abstractive rewrite", "", res.abstractive_summary, ""]
    L += ["## Structured summary", ""]
    for role, sents in res.structured.items():
        L += [f"**{role}:** " + " ".join(sents), ""]
    L += ["## Length comparison", "", res.stats_df.to_markdown(), ""]
    if res.rouge_df is not None:
        L += ["## ROUGE vs reference abstract (F1 %)", "", res.rouge_df.to_markdown(), ""]
    L += ["## Original Introduction", "", res.intro, "", "## Original Conclusion", "", res.conclusion, ""]
    return "\n".join(L)


def build_json(res) -> str:
    data = {
        "title": res.title, "source": res.source,
        "model": res.config.embedding_model, "method": res.config.method,
        "summary": res.extractive_summary, "abstractive": res.abstractive_summary,
        "structured": res.structured,
        "selected_sentences": [{"index": int(i), "section": res.section[i], "role": res.roles.get(i),
                                "text": res.sentences[i]} for i in res.selected],
        "metrics": {"coverage": round(res.coverage, 4), "keyword_retention": round(res.keyword_retention, 4),
                    "reduction_percent": float(res.stats_df.iloc[1]["Reduction (%)"]),
                    "original_words": int(res.stats_df.loc["Original", "Words"]),
                    "summary_words": int(res.stats_df.iloc[1]["Words"])},
        "rouge": None if res.rouge_df is None else res.rouge_df.to_dict(orient="index"),
        "warnings": res.warnings,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)
