"""HTML fragments for the PaperLens UI (all user text is escaped)."""
import html

ROLE_COLORS = {"Research Problem": "#ef4444", "Approach": "#3b82f6", "Major Findings": "#10b981", "Conclusion": "#f59e0b"}
ROLE_ICONS = {"Research Problem": "🎯", "Approach": "🛠️", "Major Findings": "📊", "Conclusion": "✅"}
e = html.escape


def hero_html() -> str:
    return """
<div class="pl-hero">
  <div class="pl-badge">BERT-powered &nbsp;·&nbsp; open source</div>
  <h1>🔍 PaperLens</h1>
  <p class="pl-tag">Paste a link. Get the paper's story in seconds.</p>
  <p class="pl-sub">PaperLens reads the Introduction and Conclusion of a research paper, understands every sentence
  with BERT, and returns a concise, structured and measurable summary.</p>
  <div class="pl-chips">
    <span>🔗 arXiv &amp; PDF links</span><span>🧠 BERT · SciBERT · Sentence-BERT</span>
    <span>🧩 Problem → Approach → Findings</span><span>📊 ROUGE &amp; coverage scores</span>
  </div>
</div>
<div class="pl-steps">
  <div><b>1</b><span>Fetch the paper</span></div><div><b>2</b><span>Find Intro &amp; Conclusion</span></div>
  <div><b>3</b><span>BERT embeds sentences</span></div><div><b>4</b><span>Select, label, evaluate</span></div>
</div>"""


def status_html(kind: str, title: str = "", notes=None, source: str = "") -> str:
    icon = {"ok": "✅", "error": "⚠️"}.get(kind, "ℹ️")
    head = e(title) if title else ("Summary ready" if kind == "ok" else "Something went wrong")
    src = f"<div class='pl-src'>{e(source)}</div>" if source and kind == "ok" else ""
    body = "".join(f"<li>{e(n)}</li>" for n in (notes or []))
    body = f"<ul>{body}</ul>" if body else ""
    return f"<div class='pl-status pl-{kind}'><div class='pl-st-h'>{icon} {head}</div>{src}{body}</div>"


def _card(label, value, sub=""):
    return (f"<div class='pl-card'><div class='pl-v'>{e(str(value))}</div><div class='pl-l'>{e(label)}</div>"
            f"<div class='pl-s'>{e(sub)}</div></div>")


def metric_cards(res) -> str:
    s = res.stats_df
    ow, sw = int(s.loc["Original", "Words"]), int(s.iloc[1]["Words"])
    cards = [_card("Shorter by", f"{s.iloc[1]['Reduction (%)']:.0f}%", f"{ow} → {sw} words"),
             _card("Coverage", f"{res.coverage:.2f}", "how well it represents all sentences"),
             _card("Keyword retention", f"{res.keyword_retention:.0%}", "of the top-15 keywords kept")]
    if res.rouge_df is not None:
        cards.append(_card("ROUGE-1", f"{res.rouge_df.iloc[0]['ROUGE-1']:.1f}", "vs reference abstract"))
    cards.append(_card("Time", f"{res.elapsed:.1f}s", f"{res.config.method} · {len(res.sentences)} sentences"))
    return "<div class='pl-cards'>" + "".join(cards) + "</div>"


def role_cards(res) -> str:
    out = []
    for role, sents in res.structured.items():
        c = ROLE_COLORS[role]
        out.append(f"<div class='pl-role' style='border-left-color:{c};background:{c}14'>"
                   f"<div class='pl-role-h' style='color:{c}'>{ROLE_ICONS[role]} {e(role)}</div>"
                   f"<div>{e(' '.join(sents))}</div></div>")
    return "".join(out) or "<p>No structured summary available.</p>"


def highlighted_text(res) -> str:
    legend = "".join(f"<span class='pl-chip' style='background:{c}26;border-color:{c}'>{ROLE_ICONS[r]} {r}</span>"
                     for r, c in ROLE_COLORS.items())
    blocks = [f"<div class='pl-legend'>{legend}<span class='pl-chip'>plain = not selected</span></div>"]
    for name in ("Introduction", "Conclusion"):
        idx = [i for i, s in enumerate(res.section) if s == name]
        if not idx:
            continue
        spans = []
        for i in idx:
            t = e(res.sentences[i])
            if i in res.selected:
                c = ROLE_COLORS[res.roles[i]]
                spans.append(f"<span class='pl-sel' style='background:{c}26;border-bottom:2px solid {c}' "
                             f"title='{e(res.roles[i])}'>{t}</span>")
            else:
                spans.append(f"<span class='pl-non'>{t}</span>")
        blocks.append(f"<h4>{name}</h4><div class='pl-text'>{' '.join(spans)}</div>")
    return "".join(blocks)
