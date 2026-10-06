"""Optional abstractive step: rewrite the (extracted) text with a seq2seq model.
Note: BART/DistilBART are NOT BERT models - BERT (an encoder) selects the sentences,
the seq2seq model only rewrites them fluently ("extract-then-abstract")."""

_MODELS = {}


def _load(name):
    if name not in _MODELS:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
        _MODELS[name] = (tok, model, device, torch)
    return _MODELS[name]


def _chunks(sentences, max_words=450):
    cur, n, out = [], 0, []
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def summarize_abstractive(sentences, model_name, max_words=110, min_words=35) -> str:
    tok, model, device, torch = _load(model_name)
    outs = []
    for chunk in _chunks(sentences):
        n = len(chunk.split())
        target_max = max(20, min(max_words, int(n * 0.8)))
        target_min = max(10, min(min_words, int(n * 0.3)))
        max_tokens = int(target_max * 1.4)
        min_tokens = min(int(target_min * 1.3), max_tokens - 5)
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, num_beams=4, max_length=max_tokens, min_length=min_tokens,
                                 no_repeat_ngram_size=3, early_stopping=True)
        outs.append(tok.decode(ids[0], skip_special_tokens=True).strip())
    return " ".join(outs)
