"""Command line:  paperlens <pdf | url | arXiv-id> [options]"""
import argparse

from .config import SummarizerConfig, EMBEDDING_ALIASES
from .pipeline import run_from_source, benchmark_methods
from .report import build_markdown, build_json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paperlens", description="Summarize a research paper from its Introduction and Conclusion.")
    ap.add_argument("paper", help="path to a PDF/.txt, a URL, or an arXiv id (e.g. 1810.04805)")
    ap.add_argument("--model", default="minilm", help="minilm | bert | scibert | hashing | any Hugging Face id")
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "textrank"])
    ap.add_argument("--words", type=int, default=180, help="target summary length in words")
    ap.add_argument("--abstractive", action="store_true", help="also rewrite with DistilBART")
    ap.add_argument("--reference-file", help="text file with a reference abstract (default: the paper's own abstract)")
    ap.add_argument("--benchmark", action="store_true", help="compare kmeans vs textrank")
    ap.add_argument("--out", help="write a Markdown report here")
    ap.add_argument("--json", help="write a JSON summary here")
    ap.add_argument("--plot", help="save the dashboard PNG here")
    a = ap.parse_args(argv)

    cfg = SummarizerConfig(embedding_model=EMBEDDING_ALIASES.get(a.model, a.model), method=a.method,
                           target_words=a.words, abstractive=a.abstractive)
    ref = open(a.reference_file, encoding="utf-8").read() if a.reference_file else None
    res = run_from_source(a.paper, cfg, reference=ref)

    if res.title:
        print("PAPER:", res.title)
    for w in res.warnings:
        print("NOTE:", w)
    print("\n=== SUMMARY ===\n" + res.extractive_summary)
    if res.abstractive_summary:
        print("\n=== ABSTRACTIVE REWRITE ===\n" + res.abstractive_summary)
    print("\n=== STRUCTURED ===")
    for role, s in res.structured.items():
        print(f"[{role}] " + " ".join(s))
    print("\n=== LENGTHS ===\n" + res.stats_df.to_string())
    if res.rouge_df is not None:
        print("\n=== ROUGE (F1 %) ===\n" + res.rouge_df.to_string())
    print(f"\nCoverage: {res.coverage:.3f} | Keyword retention: {res.keyword_retention:.0%} | {res.elapsed:.1f}s")
    if a.benchmark:
        print("\n=== METHOD COMPARISON ===\n" + benchmark_methods(res).to_string())
    if a.out:
        open(a.out, "w", encoding="utf-8").write(build_markdown(res)); print("Report saved:", a.out)
    if a.json:
        open(a.json, "w", encoding="utf-8").write(build_json(res)); print("JSON saved:", a.json)
    if a.plot:
        from .visualize import dashboard
        dashboard(res).savefig(a.plot, dpi=150, bbox_inches="tight"); print("Dashboard saved:", a.plot)


if __name__ == "__main__":
    main()
