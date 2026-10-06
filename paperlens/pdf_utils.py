"""Read a paper (PDF or .txt) and automatically locate Abstract, Introduction and Conclusion."""
import re
from dataclasses import dataclass, field

_CONCL = (r"(?:conclusions?(?:\s+and\s+(?:future\s+work|discussion|outlook|limitations))?"
          r"|concluding\s+remarks|discussion\s+and\s+conclusions?|summary\s+and\s+conclusions?"
          r"|summary(?:\s+and\s+future\s+work)?|future\s+work)")
_OTHER = (r"(?:background|related\s+work|literature\s+review|preliminaries|methods?|methodology"
          r"|materials\s+and\s+methods|proposed\s+(?:method|approach|model)|approach|experiments?"
          r"|experimental\s+(?:setup|results|evaluation)|results?|results\s+and\s+discussion|evaluation"
          r"|discussion|analysis|limitations|acknowledge?ments?|references|bibliography|appendix(?:\s+[a-z])?)")
_NUM = r"(?:(?:\d{1,2}|[IVXL]{1,5})[\.\)]?\s+)?"
_NAMED = re.compile(r"^" + _NUM + r"(?P<name>abstract|introduction|" + _CONCL + "|" + _OTHER + r")\s*[:.]?$", re.I)
_INLINE_ABS = re.compile(r"^abstract\s*[\u2014\u2013:\.\-]+\s*(?P<rest>\S.*)$", re.I)
_GENERIC = re.compile(r"^(?:\d{1,2}|[IVX]{1,4})[\.\)]?\s+[A-Z][A-Za-z0-9\-:,&' ]{2,60}$")
_SUBSECTION = re.compile(r"^\d+(?:\.\d+)+\.?\s+[A-Z][^.]{0,60}$")
_NUM_ONLY = re.compile(r"^(?:\d{1,2}|[IVXL]{1,5})[\.\)]?$")


@dataclass
class Sections:
    abstract: str = ""
    introduction: str = ""
    conclusion: str = ""
    title: str = ""
    method: str = "headings"          # "headings", "partial-fallback" or "fallback"
    warnings: list = field(default_factory=list)


def extract_text_from_pdf(source) -> str:
    from pypdf import PdfReader
    reader = PdfReader(source)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def pdf_title(source, text: str = "") -> str:
    """Best-effort paper title: PDF metadata, else the first reasonable line of text."""
    try:
        from pypdf import PdfReader
        t = (PdfReader(source).metadata.title or "").strip()
        if len(t) >= 12 and not t.lower().startswith(("microsoft word", "untitled")) and "." not in t[-5:-1]:
            return t[:200]
    except Exception:
        pass
    for line in text.splitlines():
        line = line.strip()
        if 15 <= len(line) <= 200:
            return line
    return ""


def _kind(name: str) -> str:
    n = re.sub(r"\s+", " ", name.lower())
    if n.startswith("abstract"):
        return "abstract"
    if n.startswith("introduction"):
        return "intro"
    if re.fullmatch(_CONCL, n, re.I):
        return "conclusion"
    if n.startswith(("references", "bibliography", "acknowledg")):
        return "end"
    return "other"


def _classify(line: str):
    m = _INLINE_ABS.match(line)
    if m:
        return ("abstract", m.group("rest"))
    if len(line) > 80:
        return None
    m = _NAMED.match(line)
    if m:
        return (_kind(m.group("name")), "")
    if _GENERIC.match(line) and len(line.split()) <= 7:
        return ("other", "")
    return None


def _prepare_lines(text: str) -> list:
    raw = [l.strip() for l in text.replace("\r", "").split("\n")]
    raw = [l for l in raw if l]
    out, i = [], 0
    while i < len(raw):
        l = raw[i]
        if _NUM_ONLY.match(l):
            nxt = raw[i + 1] if i + 1 < len(raw) else ""
            if nxt and _classify(nxt):
                out.append(l + " " + nxt)
                i += 2
                continue
            if re.fullmatch(r"\d{1,3}", l):      # stray page number
                i += 1
                continue
        out.append(l)
        i += 1
    return out


def find_sections(text: str, min_intro_words: int = 40, min_concl_words: int = 25,
                  max_words: int = 2500) -> Sections:
    lines = _prepare_lines(text)
    heads = []
    for i, l in enumerate(lines):
        c = _classify(l)
        if c:
            heads.append((i, c[0], c[1], l))

    def body(p):
        i, _, inline, _ = heads[p]
        end = heads[p + 1][0] if p + 1 < len(heads) else len(lines)
        parts = ([inline] if inline else []) + [l for l in lines[i + 1:end] if not _SUBSECTION.match(l)]
        return " ".join(" ".join(parts).split()[:max_words])

    res = Sections()
    for p, h in enumerate(heads):
        if h[1] == "abstract" and len(body(p).split()) >= 20:
            res.abstract = body(p)
            break
    for p, h in enumerate(heads):
        if h[1] == "intro" and len(body(p).split()) >= min_intro_words:
            res.introduction = body(p)
            break
    cands = [(p, h) for p, h in enumerate(heads) if h[1] == "conclusion"]
    cands.sort(key=lambda x: ("conclu" not in x[1][3].lower(), -x[0]))   # prefer "Conclusion", then later ones
    for p, _ in cands:
        if len(body(p).split()) >= min_concl_words:
            res.conclusion = body(p)
            break

    # ---- fallbacks when headings were not found ----
    refs = [h[0] for h in heads if h[1] == "end" and re.match(r"^(?:\S+\s+)?(references|bibliography)", h[3], re.I)]
    end_line = refs[-1] if refs else len(lines)
    words = " ".join(lines[:end_line]).split()
    found = int(bool(res.introduction)) + int(bool(res.conclusion))
    if not res.introduction:
        skip = len(res.abstract.split()) if res.abstract else 0
        n = min(600, max(150, int(0.12 * len(words))))
        res.introduction = " ".join(words[skip:skip + n])
        res.warnings.append("Introduction heading not found - used the first part of the document instead.")
    if not res.conclusion:
        n = min(500, max(120, int(0.08 * len(words))))
        res.conclusion = " ".join(words[-n:])
        res.warnings.append("Conclusion heading not found - used the last part of the document instead.")
    res.method = "headings" if found == 2 else ("partial-fallback" if found == 1 else "fallback")
    if not res.abstract:
        res.warnings.append("No abstract detected - ROUGE evaluation needs a reference summary (paste one manually).")
    return res


def load_sections(path: str) -> Sections:
    is_pdf = str(path).lower().endswith(".pdf")
    if is_pdf:
        text = extract_text_from_pdf(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    if len(text.split()) < 100:
        raise ValueError("Very little text was extracted. Scanned PDFs need OCR first (e.g. ocrmypdf).")
    sec = find_sections(text)
    sec.title = pdf_title(path, text) if is_pdf else next((l.strip() for l in text.splitlines() if l.strip()), "")[:200]
    return sec
