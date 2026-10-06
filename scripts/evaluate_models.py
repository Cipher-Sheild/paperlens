"""Benchmark models/methods over many papers and print one comparison table.

  python scripts/evaluate_models.py --pdf-dir papers/ --models hashing,minilm,bert,scibert --methods kmeans,textrank
  python scripts/evaluate_models.py --sources list.txt --models hashing,minilm      # one path/URL/arXiv-id per line

ROUGE is computed against each paper's own abstract (papers without a detectable abstract are skipped for ROUGE).
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from paperlens import SummarizerConfig, EMBEDDING_ALIASES, run_pipeline
from paperlens.fetch import download_pdf
from paperlens.pdf_utils import load_sections


def gather(args):
    srcs = []
    if args.pdf_dir:
        srcs += sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if args.sources:
        srcs += [l.strip() for l in open(args.sources, encoding="utf-8") if l.strip() and not l.startswith("#")]
    return srcs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir"); ap.add_argument("--sources")
    ap.add_argument("--models", default="hashing,minilm,bert")
    ap.add_argument("--methods", default="kmeans,textrank")
    ap.add_argument("--words", type=int, default=180)
    ap.add_argument("--out", default="benchmark_results.csv")
    ap.add_argument("--plot")
    a = ap.parse_args(argv)

    papers = []
    for s in gather(a):
        try:
            path = s if os.path.exists(s) else download_pdf(s)
            papers.append((s, load_sections(path)))
        except Exception as e:
            print(f"skip {s}: {e}")
    if not papers:
        sys.exit("No papers could be loaded.")

    rows = []
    for m in a.models.split(","):
        model_id = EMBEDDING_ALIASES.get(m, m)
        for method in a.methods.split(","):
            cfg = SummarizerConfig(embedding_model=model_id, method=method, target_words=a.words)
            for src, sec in papers:
                try:
                    r = run_pipeline(sec.introduction, sec.conclusion, cfg, sec.abstract or None, title=sec.title)
                except Exception as e:
                    print(f"fail {m}/{method}/{src}: {e}"); continue
                row = {"model": m, "method": method, "paper": os.path.basename(src), "words": int(r.stats_df.iloc[1]["Words"]),
                       "reduction_%": float(r.stats_df.iloc[1]["Reduction (%)"]), "coverage": r.coverage,
                       "keyword_retention": r.keyword_retention}
                if r.rouge_df is not None:
                    row.update({k: v for k, v in r.rouge_df.iloc[0].items()})
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    summary = df.groupby(["model", "method"]).mean(numeric_only=True).round(3)
    summary["papers"] = df.groupby(["model", "method"]).size()
    print("\n=== MEAN RESULTS OVER", len(papers), "PAPER(S) ===")
    print(summary.to_string())
    print("\nPer-paper results saved to", a.out)
    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = [c for c in ("ROUGE-1", "ROUGE-2", "ROUGE-L", "coverage") if c in summary.columns]
        ax = summary[cols].plot(kind="bar", figsize=(11, 5))
        ax.set_title("PaperLens benchmark (mean over papers)"); ax.set_ylabel("score")
        plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(a.plot, dpi=150)
        print("Plot saved to", a.plot)
    return summary


if __name__ == "__main__":
    main()
