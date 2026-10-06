# Changelog

## 1.1.1
**Fixed**
- A link left in the *Paper link* box no longer overrides an uploaded PDF: the **active tab** now decides the input.
- DOIs (e.g. `10.1109/...`) were mistaken for IP addresses and rejected as "private/internal". DOIs are now resolved through doi.org; bare strings must look like real domains.
- Clear messages for publisher sites that need a login (IEEE Xplore, ScienceDirect, ACM, Wiley, Springer...): download the PDF and use *Upload PDF*.

## 1.1.0 - PaperLens
**New**
- Project renamed **PaperLens**; installable package (`pip install -e .`) with `paperlens` CLI and `paperlens-app`.
- **Paper links**: paste a direct PDF link, an arXiv URL or a bare arXiv id. Landing pages that expose `citation_pdf_url` also work. SSRF protection (private/internal addresses and redirects are blocked), 30 MB limit.
- Redesigned web UI: hero header, metric cards, colour-coded Problem/Approach/Findings/Conclusion cards, highlighted-paper view, progress messages, JSON + Markdown export.
- Multi-paper benchmark script `scripts/evaluate_models.py` (models x methods x papers -> table/CSV/plot).
- Paper title detection.

**Improved**
- PDF cleaning: repaired ligature splits ("identifi ed"), "high -quality", ".." and stray spaces; removed URLs and table/formula junk.
- **Mean-centered embeddings** (sharper similarities) and a centrality-based importance score.
- **Length-targeted selection** with a guaranteed share for the Conclusion (default 60/40) instead of a fixed ratio.
- **Sentence-quality scoring** prefers self-contained, informative sentences over dangling ("However, it ...") or trivia sentences.
- Structure labels now use BERT similarity to prototype sentences (works for review papers too).
- Default encoder: Sentence-BERT MiniLM (BERT-base remains one click away).

## 1.0.0
- First version: BERT embeddings + K-Means/TextRank, PDF section detection, ROUGE, dashboard, tests.
