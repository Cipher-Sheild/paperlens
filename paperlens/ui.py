"""PaperLens web app (Gradio)."""
import argparse
import inspect
import os
import tempfile
import time

import gradio as gr
import pandas as pd

from .config import SummarizerConfig, EMBEDDING_MODELS, ABSTRACTIVE_MODELS
from .fetch import download_pdf
from .html_views import hero_html, status_html, metric_cards, role_cards, highlighted_text
from .pdf_utils import load_sections
from .pipeline import run_pipeline, benchmark_methods
from .report import build_markdown, build_json
from .sample import SAMPLE_INTRO, SAMPLE_CONCLUSION, SAMPLE_ABSTRACT
from .visualize import dashboard

CSS = """
.gradio-container {max-width: 1400px !important; margin: auto;}
.pl-hero {background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%); color:#fff; border-radius:20px;
  padding:34px 38px; box-shadow:0 10px 30px rgba(79,70,229,.35);}
.pl-hero h1 {font-size:2.7rem; margin:10px 0 4px; color:#fff; letter-spacing:-.5px;}
.pl-badge {display:inline-block; background:rgba(255,255,255,.18); padding:4px 12px; border-radius:999px; font-size:.8rem;}
.pl-tag {font-size:1.3rem; font-weight:600; margin:0 0 6px; color:#fff;}
.pl-sub {max-width:780px; opacity:.92; margin:0 0 16px; color:#fff;}
.pl-chips span {display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);
  padding:5px 12px; border-radius:999px; font-size:.85rem; margin:0 8px 8px 0;}
.pl-steps {display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:14px 0 6px;}
.pl-steps div {display:flex; align-items:center; gap:10px; padding:10px 14px; border-radius:12px;
  border:1px solid var(--border-color-primary); background:var(--block-background-fill);}
.pl-steps b {background:#6366f1; color:#fff; width:26px; height:26px; border-radius:50%; display:flex;
  align-items:center; justify-content:center; font-size:.85rem; flex:none;}
.pl-status {border-radius:14px; padding:14px 18px; border:1px solid var(--border-color-primary); margin-bottom:8px;}
.pl-ok {border-left:6px solid #10b981; background:rgba(16,185,129,.08);}
.pl-error {border-left:6px solid #ef4444; background:rgba(239,68,68,.10);}
.pl-st-h {font-weight:700; font-size:1.05rem;} .pl-src {opacity:.7; font-size:.85rem; word-break:break-all;}
.pl-status ul {margin:8px 0 0 18px; padding:0; font-size:.9rem;}
.pl-cards {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:6px 0 12px;}
.pl-card {border:1px solid var(--border-color-primary); background:var(--block-background-fill); border-radius:14px;
  padding:14px 16px; text-align:center;}
.pl-v {font-size:1.9rem; font-weight:800; background:linear-gradient(135deg,#6366f1,#ec4899);
  -webkit-background-clip:text; background-clip:text; color:transparent;}
.pl-l {font-weight:600; font-size:.85rem; text-transform:uppercase; letter-spacing:.04em;}
.pl-s {font-size:.75rem; opacity:.65; margin-top:2px;}
.pl-role {border-left:5px solid; border-radius:10px; padding:12px 16px; margin:10px 0; line-height:1.55;}
.pl-role-h {font-weight:700; margin-bottom:4px;}
.pl-legend {margin:4px 0 10px;} .pl-chip {display:inline-block; border:1px solid var(--border-color-primary);
  padding:3px 10px; border-radius:999px; font-size:.8rem; margin:0 6px 6px 0;}
.pl-text {line-height:1.85; font-size:.97rem;} .pl-sel {padding:2px 3px; border-radius:4px;} .pl-non {opacity:.62;}
#run-btn {background:linear-gradient(135deg,#4f46e5,#db2777) !important; color:#fff !important; border:none !important;
  font-weight:700; font-size:1.05rem; box-shadow:0 6px 18px rgba(124,58,237,.35);}
.pl-foot {text-align:center; opacity:.65; font-size:.85rem; margin-top:14px;}
"""

try:
    THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate")
except Exception:                                   # very old/new Gradio
    THEME = None

_BLOCKS_TAKES_STYLE = "css" in inspect.signature(gr.Blocks.__init__).parameters     # Gradio <= 5
_STYLE = {k: v for k, v in (("theme", THEME), ("css", CSS)) if v is not None}


def _error(msg):
    return (status_html("error", notes=[msg]), "", "", "", "", "", None,
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None)


def process(mode, url, file, intro_txt, concl_txt, ref_txt, emb_label, method, target_words, concl_share,
            use_quality, use_abs, abs_label, do_bench, progress=gr.Progress()):
    t0 = time.perf_counter()
    try:
        cfg = SummarizerConfig(embedding_model=EMBEDDING_MODELS[emb_label], method=method,
                               target_words=int(target_words), conclusion_share=float(concl_share),
                               use_quality=bool(use_quality), abstractive=bool(use_abs),
                               abstractive_model=ABSTRACTIVE_MODELS[abs_label])
        reference = (ref_txt or "").strip() or None
        notes, title, source = [], "", "pasted text"

        mode, url = (mode or "link"), (url or "").strip()
        path = None
        if mode == "link":                      # the tab you are on decides what is used
            if url:
                progress(0.05, desc="Downloading paper...")
                path, source = download_pdf(url), url
            elif file:
                path = file if isinstance(file, str) else file.name
                source = os.path.basename(path)
            else:
                raise ValueError("Paste a paper link first - or switch to the 'Upload PDF' / 'Paste text' tab.")
        elif mode == "file":
            if not file:
                raise ValueError("Upload a PDF first - or switch to the 'Paper link' / 'Paste text' tab.")
            path = file if isinstance(file, str) else file.name
            source = os.path.basename(path)

        if path:
            progress(0.2, desc="Finding Introduction & Conclusion...")
            sec = load_sections(path)
            intro, concl, title = sec.introduction, sec.conclusion, sec.title
            reference = reference or sec.abstract or None
            notes += sec.warnings
            if sec.method == "headings":
                notes.append("Introduction and Conclusion found from the paper's headings.")
        else:
            intro, concl = intro_txt or "", concl_txt or ""

        progress(0.4, desc="BERT is reading the sentences (first run downloads the model)...")
        res = run_pipeline(intro, concl, cfg, reference, title=title, source=source)
        res.elapsed = time.perf_counter() - t0
        notes += res.warnings

        progress(0.9, desc="Drawing charts...")
        stats = res.stats_df.reset_index().rename(columns={"index": "Text"})
        rouge = (res.rouge_df.reset_index().rename(columns={"index": "Summary"}) if res.rouge_df is not None
                 else pd.DataFrame({"Info": ["No reference abstract found - paste one under 'Optional' to get ROUGE."]}))
        bench = benchmark_methods(res).reset_index().rename(columns={"index": "Method"}) if do_bench else pd.DataFrame()

        out_dir = tempfile.mkdtemp()
        md_path, js_path = os.path.join(out_dir, "paperlens_report.md"), os.path.join(out_dir, "paperlens_summary.json")
        open(md_path, "w", encoding="utf-8").write(build_markdown(res))
        open(js_path, "w", encoding="utf-8").write(build_json(res))

        return (status_html("ok", title or "Summary ready", notes, source), metric_cards(res),
                res.extractive_summary, role_cards(res), res.abstractive_summary or "", highlighted_text(res),
                dashboard(res), stats, rouge, bench, [md_path, js_path])
    except Exception as e:
        return _error(str(e))


def build_ui():
    kw = {"title": "PaperLens - BERT research paper summarizer"}
    if _BLOCKS_TAKES_STYLE:
        kw.update(_STYLE)
    with gr.Blocks(**kw) as demo:
        gr.HTML(hero_html())
        with gr.Row():
            with gr.Column(scale=4, min_width=360):
                mode = gr.State("link")
                with gr.Tabs():
                    with gr.Tab("🔗 Paper link") as tab_link:
                        url = gr.Textbox(label="Paper URL or arXiv ID", placeholder="arXiv id (1810.04805), arXiv/PDF link, or an open-access DOI")
                        gr.Examples([["1706.03762"], ["1810.04805"], ["1908.10084"]], inputs=[url],
                                    label="Try a famous paper (arXiv)", cache_examples=False)
                    with gr.Tab("📄 Upload PDF") as tab_file:
                        file = gr.File(label="PDF or .txt", file_types=[".pdf", ".txt"], type="filepath")
                    with gr.Tab("✍️ Paste text") as tab_text:
                        intro = gr.Textbox(label="Introduction", value=SAMPLE_INTRO, lines=6)
                        concl = gr.Textbox(label="Conclusion", value=SAMPLE_CONCLUSION, lines=5)
                tab_link.select(lambda: "link", None, mode)
                tab_file.select(lambda: "file", None, mode)
                tab_text.select(lambda: "text", None, mode)
                gr.Markdown("<small>The tab you are on decides what is summarized. Publisher sites that need a login "
                            "(IEEE, ScienceDirect, ACM...) can't be downloaded - save the PDF and use <b>Upload PDF</b>.</small>")
                with gr.Accordion("Optional: reference abstract (for ROUGE)", open=False):
                    ref = gr.Textbox(label="Reference abstract", value=SAMPLE_ABSTRACT, lines=3,
                                     info="Auto-detected from PDFs. Used for the sample text only if nothing else is provided.")
                with gr.Accordion("⚙️ Settings", open=False):
                    emb = gr.Dropdown(list(EMBEDDING_MODELS), value=list(EMBEDDING_MODELS)[0], label="BERT model")
                    method = gr.Radio(["kmeans", "textrank"], value="kmeans", label="Selection method")
                    words = gr.Slider(60, 400, value=180, step=10, label="Target summary length (words)")
                    share = gr.Slider(0.2, 0.6, value=0.4, step=0.05, label="Share of summary from the Conclusion")
                    quality = gr.Checkbox(value=True, label="Prefer self-contained, informative sentences")
                    use_abs = gr.Checkbox(label="Also rewrite with an abstractive model (slower, extra download)")
                    abs_model = gr.Dropdown(list(ABSTRACTIVE_MODELS), value=list(ABSTRACTIVE_MODELS)[0], label="Abstractive model")
                    bench = gr.Checkbox(label="Compare K-Means vs TextRank")
                btn = gr.Button("✨ Summarize", elem_id="run-btn", variant="primary", size="lg")
            with gr.Column(scale=7, min_width=480):
                status = gr.HTML(status_html("info", "Ready", ["Choose a paper on the left and press Summarize. "
                                                               "The first run downloads the BERT model (about a minute)."]))
                metrics = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("✨ Summary"):
                        summary = gr.Textbox(label="Extractive summary (BERT)", lines=7, interactive=False)
                        roles = gr.HTML()
                        abstractive = gr.Textbox(label="Abstractive rewrite (optional)", lines=4, interactive=False)
                    with gr.Tab("🖍️ Highlighted paper"):
                        highlight = gr.HTML()
                    with gr.Tab("📊 Dashboard"):
                        plot = gr.Plot(label="Dashboard")
                    with gr.Tab("📈 Evaluation"):
                        gr.Markdown("**Length comparison**")
                        stats = gr.Dataframe(interactive=False)
                        gr.Markdown("**ROUGE (F1 %) against the reference abstract**")
                        rouge = gr.Dataframe(interactive=False)
                        gr.Markdown("**K-Means vs TextRank**")
                        benchdf = gr.Dataframe(interactive=False)
                    with gr.Tab("📥 Export"):
                        files = gr.File(label="Report (.md) and data (.json)", file_count="multiple")
        gr.HTML("<div class='pl-foot'>PaperLens v1.1 · built with BERT, scikit-learn and Gradio · "
                "summaries are extractive: every sentence comes from the paper.</div>")
        btn.click(process, [mode, url, file, intro, concl, ref, emb, method, words, share, quality, use_abs, abs_model, bench],
                  [status, metrics, summary, roles, abstractive, highlight, plot, stats, rouge, benchdf, files])
    return demo


def launch(demo, **kwargs):
    if not _BLOCKS_TAKES_STYLE:
        kwargs.update(_STYLE)
    demo.launch(**kwargs)


def main():
    ap = argparse.ArgumentParser(description="PaperLens web app")
    ap.add_argument("--share", action="store_true", help="create a public link (useful in Colab)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    launch(build_ui(), share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
