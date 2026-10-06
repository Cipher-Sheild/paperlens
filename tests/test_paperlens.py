import functools
import http.server
import os
import tempfile
import threading

import numpy as np
import pytest

from paperlens import SummarizerConfig, run_pipeline, run_from_source, benchmark_methods
from paperlens import sample as S
from paperlens.embedder import HashingEmbedder
from paperlens.evaluate import rouge_table
from paperlens.extractive import select_kmeans, select_textrank, center_embeddings, sentences_needed, coverage_score
from paperlens.fetch import normalize_source, download_pdf, paywall_hint
from paperlens.html_views import highlighted_text, metric_cards, role_cards
from paperlens.pdf_utils import find_sections
from paperlens.preprocess import clean_text, split_sentences
from paperlens.report import build_json, build_markdown
from paperlens.scoring import quality_score
from paperlens.visualize import dashboard

CFG = SummarizerConfig(embedding_model="hashing")


# ---------- cleaning ----------
def test_cleaning_fixes_pdf_glitches():
    t = clean_text("The study identifi ed key topics in high -quality work at the Conference.. See ( Fig. 2 ) and https://x.org/a.")
    assert "identified" in t and "high-quality" in t and "Conference." in t and ".." not in t
    assert "https" not in t and "( " not in t
    assert clean_text("the \ufb01rst \ufb02ow") == "the first flow"


def test_split_sentences_protects_abbreviations_and_drops_junk():
    s = split_sentences("Recent work [1, 2] shows gains, e.g. in vision. Smith et al. reported it clearly in earlier experiments. "
                        "1 2 3 4 5 6 7 8 9 0 1 2. Fig. 3 shows more results here.")
    assert len(s) == 3 and all("[1" not in x for x in s)


# ---------- sections ----------
def test_find_sections_text():
    text = (f"Title\nAbstract\n{S.ABSTRACT}\n1 Introduction\n{S.INTRO_1} {S.INTRO_2}\n1.1 Contributions\n{S.INTRO_3}\n"
            f"2 Related Work\n{'filler words here. ' * 30}\n5 Conclusion\n{S.CONCLUSION}\nReferences\n[1] x")
    sec = find_sections(text)
    assert sec.method == "headings" and not sec.warnings
    assert "Contributions" not in sec.introduction and sec.conclusion.startswith("In this work")


def test_find_sections_split_number_lines_and_fallback():
    text = f"Abstract\n{S.ABSTRACT}\n1\nIntroduction\n{S.SAMPLE_INTRO}\n2\nMethod\n{'x y z. ' * 30}\n3\nConclusions\n{S.CONCLUSION}\n"
    assert find_sections(text).method == "headings"
    assert find_sections("word " * 1000).method == "fallback"


def test_pdf_roundtrip_and_title():
    pytest.importorskip("reportlab"); pytest.importorskip("pypdf")
    from scripts.make_sample_pdf import build
    from paperlens.pdf_utils import load_sections
    p = os.path.join(tempfile.mkdtemp(), "p.pdf"); build(p)
    sec = load_sections(p)
    assert sec.method == "headings" and sec.abstract and "Summarizing Research Papers" in sec.title


# ---------- fetching ----------
@pytest.mark.parametrize("src,expected", [
    ("1706.03762", "https://arxiv.org/pdf/1706.03762"),
    ("arXiv:1706.03762v5", "https://arxiv.org/pdf/1706.03762"),
    ("https://arxiv.org/abs/1810.04805v2", "https://arxiv.org/pdf/1810.04805"),
    ("https://arxiv.org/pdf/1908.10084.pdf", "https://arxiv.org/pdf/1908.10084"),
    ("hep-th/9901001", None),
    ("10.1109", None),
    ("10.1109/GSEACT68539.2026.11620474", "https://doi.org/10.1109/GSEACT68539.2026.11620474"),
    ("doi:10.1038/nature14539", "https://doi.org/10.1038/nature14539"),
    ("https://doi.org/10.1145/3292500.3330701", "https://doi.org/10.1145/3292500.3330701"),
    ("example.org/paper.pdf", "https://example.org/paper.pdf"),
])
def test_normalize_source(src, expected):
    if expected is None:
        with pytest.raises(ValueError):
            normalize_source(src)
    else:
        assert normalize_source(src) == expected


def test_paywall_hint():
    assert "IEEE Xplore" in paywall_hint("https://ieeexplore.ieee.org/document/11620474")
    assert "Upload PDF" in paywall_hint("https://www.sciencedirect.com/science/article/pii/X")
    assert "IEEE Xplore" in paywall_hint("https://doi.org/10.1109/GSEACT68539.2026.11620474")
    assert paywall_hint("https://arxiv.org/pdf/1706.03762") == ""


def test_normalize_rejects_garbage():
    for bad in ("", "not a link at all"):
        with pytest.raises(ValueError):
            normalize_source(bad)


@pytest.fixture(scope="module")
def local_server():
    pytest.importorskip("reportlab")
    from scripts.make_sample_pdf import build
    d = tempfile.mkdtemp()
    build(os.path.join(d, "paper.pdf"))
    open(os.path.join(d, "landing.html"), "w").write(
        '<html><head><meta name="citation_pdf_url" content="/paper.pdf"></head><body>hi</body></html>')
    open(os.path.join(d, "plain.html"), "w").write("<html><body>no pdf here</body></html>")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_private_addresses_blocked(local_server):
    with pytest.raises(ValueError, match="private"):
        download_pdf(local_server + "/paper.pdf")


def test_download_direct_and_via_landing_page(local_server):
    for path in ("/paper.pdf", "/landing.html"):
        f = download_pdf(local_server + path, allow_private=True)
        assert open(f, "rb").read(4) == b"%PDF"


def test_download_non_pdf_error(local_server):
    with pytest.raises(ValueError, match="did not return a PDF"):
        download_pdf(local_server + "/plain.html", allow_private=True)


def test_run_from_source_url(local_server):
    r = run_from_source(local_server + "/paper.pdf", CFG, allow_private=True)
    assert r.title and r.reference and r.selected


# ---------- selection ----------
def test_quality_prefers_self_contained_sentences():
    good = "In this paper, we propose a new approach for summarizing long documents automatically."
    bad = "However, it highlights how crucial it is to handle issues."
    trivia = "John McCarthy first used the phrase in 1956 at the Dartmouth Conference in New Hampshire."
    assert quality_score(good) > 0 > quality_score(bad)
    assert quality_score(trivia) < 0


def test_centering_increases_contrast():
    rng = np.random.default_rng(0)
    common = rng.normal(size=(1, 64)) * 5                      # shared direction, like raw BERT
    E = common + rng.normal(size=(30, 64))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    before = (E @ E.T)[np.triu_indices(30, 1)].mean()
    C = center_embeddings(E)
    after = (C @ C.T)[np.triu_indices(30, 1)].mean()
    assert before > 0.8 > after


def test_selectors_and_budget():
    sents = split_sentences(S.SAMPLE_INTRO)
    E = HashingEmbedder().encode(sents)
    idx = list(range(len(E)))
    for f in (select_kmeans, select_textrank):
        out = f(E, idx, 3)
        assert out == sorted(out) and 1 <= len(out) <= 3
    assert sentences_needed(sents, idx, 10) == 1
    assert sentences_needed(sents, idx, 10_000) == len(idx)


# ---------- pipeline ----------
def test_pipeline_end_to_end_and_balance():
    r = run_pipeline(S.SAMPLE_INTRO, S.SAMPLE_CONCLUSION, CFG, reference=S.ABSTRACT)
    secs = {r.section[i] for i in r.selected}
    assert secs == {"Introduction", "Conclusion"}
    assert 60 <= len(r.extractive_summary.split()) <= 300
    assert r.rouge_df.shape[1] == 3 and 0 < r.coverage <= 1
    assert set(r.roles) == set(r.selected) and r.structured
    assert benchmark_methods(r).shape[0] == 2
    assert len(dashboard(r).axes) >= 6


def test_conclusion_share_controls_balance():
    small = run_pipeline(S.SAMPLE_INTRO, S.SAMPLE_CONCLUSION, SummarizerConfig(embedding_model="hashing", conclusion_share=0.2))
    big = run_pipeline(S.SAMPLE_INTRO, S.SAMPLE_CONCLUSION, SummarizerConfig(embedding_model="hashing", conclusion_share=0.6))
    cnt = lambda r: sum(r.section[i] == "Conclusion" for i in r.selected)
    assert cnt(big) >= cnt(small)


def test_textrank_and_options():
    r = run_pipeline(S.SAMPLE_INTRO, S.SAMPLE_CONCLUSION,
                     SummarizerConfig(embedding_model="hashing", method="textrank", use_quality=False, center_embeddings=False))
    assert r.selected


def test_errors():
    with pytest.raises(ValueError):
        run_pipeline("Short text only here.", "", CFG)
    for kw in ({"method": "nope"}, {"target_words": 5}, {"conclusion_share": 0.95}):
        with pytest.raises(ValueError):
            SummarizerConfig(**kw)


# ---------- exports & views ----------
def test_reports_and_html_escaping():
    import json
    r = run_pipeline(S.SAMPLE_INTRO, S.SAMPLE_CONCLUSION, CFG, reference=S.ABSTRACT, title="A <b>title</b>")
    assert "Summary" in build_markdown(r)
    data = json.loads(build_json(r))
    assert data["metrics"]["summary_words"] > 0 and data["selected_sentences"]
    r.sentences[0] = "<script>alert(1)</script> is a dangerous looking sentence for testing."
    for html_out in (highlighted_text(r), role_cards(r), metric_cards(r)):
        assert "<script>" not in html_out


def test_ui_process_and_build():
    import paperlens.ui as ui
    ui.build_ui()
    emb = list(ui.EMBEDDING_MODELS)[-1]
    out = ui.process("text", "", None, S.SAMPLE_INTRO, S.SAMPLE_CONCLUSION, S.ABSTRACT, emb, "kmeans", 180, 0.4, True, False,
                     list(ui.ABSTRACTIVE_MODELS)[0], True)
    assert "pl-ok" in out[0] and len(out) == 11 and out[6] is not None and len(out[10]) == 2
    bad = ui.process("link", "not a link", None, "", "", "", emb, "kmeans", 180, 0.4, True, False, list(ui.ABSTRACTIVE_MODELS)[0], False)
    assert "pl-error" in bad[0]


def test_active_tab_decides_input(tmp_path):
    """Regression: a leftover URL in the link box must NOT override an uploaded PDF."""
    pytest.importorskip("reportlab")
    import paperlens.ui as ui
    from scripts.make_sample_pdf import build
    pdf = str(tmp_path / "up.pdf"); build(pdf)
    emb, ab = list(ui.EMBEDDING_MODELS)[-1], list(ui.ABSTRACTIVE_MODELS)[0]
    ok = ui.process("file", "https://ieeexplore.ieee.org/document/11620474", pdf, "", "", "", emb, "kmeans", 180, 0.4, True, False, ab, False)
    assert "pl-ok" in ok[0] and "up.pdf" in ok[0]
    nofile = ui.process("file", "", None, "", "", "", emb, "kmeans", 180, 0.4, True, False, ab, False)
    assert "Upload a PDF first" in nofile[0]
    nolink = ui.process("link", "", None, "", "", "", emb, "kmeans", 180, 0.4, True, False, ab, False)
    assert "Paste a paper link" in nolink[0]


def test_evaluate_script(tmp_path):
    pytest.importorskip("reportlab")
    from scripts.make_sample_pdf import build
    from scripts.evaluate_models import main
    build(str(tmp_path / "a.pdf")); build(str(tmp_path / "b.pdf"))
    summary = main(["--pdf-dir", str(tmp_path), "--models", "hashing", "--out", str(tmp_path / "r.csv")])
    assert summary.loc[("hashing", "kmeans"), "papers"] == 2 and (tmp_path / "r.csv").exists()
