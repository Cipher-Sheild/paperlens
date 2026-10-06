"""Text cleaning and sentence splitting (with fixes for common PDF-extraction glitches)."""
import re

STOPWORDS = set("""a an the and or of to in on for with by is are was were be been that this these those it its as at from
we our which can more than also such not but their they has have had using used use one two both each other into
most only over any all may so if then there while between however when what how why who whom whose could should would
will shall might must do does did done being very much many some few several about above below under again further
once here per via within without across among upon""".split())

_ABBREV = ["et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Eq.", "Eqs.", "Sec.", "vs.", "cf.",
           "Dr.", "Prof.", "No.", "approx.", "etc."]
_CITE_NUM = re.compile(r"\s*\[\s*\d+(?:\s*[,\u2013\u2014-]\s*\d+)*\s*\]")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
_SPLIT_LIG = re.compile(r"\b(\w{2,}(?:fi|fl|ffi|ffl)) ([a-z]+)\b")
_NOT_SPLIT = {"wifi", "hifi", "scifi"}


def _join_split_ligature(m):
    left = m.group(1)
    return m.group(0) if left.lower() in _NOT_SPLIT else left + m.group(2)


def clean_text(text: str) -> str:
    """Fix ligatures ('identifi ed' -> 'identified'), hyphenation, citations, stray spaces, double periods."""
    text = text.replace("\u00ad", "")
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    text = _CITE_NUM.sub("", text)
    text = _URL.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _SPLIT_LIG.sub(_join_split_ligature, text)
    text = re.sub(r"(?<=\w) -(?=\w)", "-", text)              # "high -quality" -> "high-quality"
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)            # "Conference.." -> "Conference."
    text = re.sub(r"\s+([,;:.!?])(?=\s|$)", r"\1", text)      # "word ," -> "word,"
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def _looks_like_text(s: str) -> bool:
    letters = sum(c.isalpha() for c in s)
    return letters / max(1, len(s)) >= 0.6          # drops table rows, formulas, numeric junk


def split_sentences(text: str, min_words: int = 6) -> list:
    """Split into clean sentences, protecting abbreviations (et al., e.g., Fig. ...)."""
    t = clean_text(text)
    for a in _ABBREV:
        t = t.replace(a, a.replace(".", "<DOT>"))
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c(\[])', t)
    out, seen = [], set()
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        key = p.lower()
        if len(p.split()) >= min_words and key not in seen and _looks_like_text(p):
            seen.add(key)
            out.append(p)
    return out


def word_count(text: str) -> int:
    return len(text.split())
