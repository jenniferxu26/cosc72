"""
python 5b_clean_preds.py infer/baseline_pred.jsonl infer/baseline_pred_clean.jsonl

python 5b_clean_preds.py infer/stacked_pred.jsonl  infer/stacked_pred_clean.jsonl
"""

import json, re, sys, pathlib

# config
TAG_RE   = re.compile(r"</?[^>]+?>")          # strip XML/HTML
URL_RE   = re.compile(r"https?://\S+")
MD_RE    = re.compile(r"^\s*(?:#{1,6}|[-*]|>\s|\d+\.)[^\n]*$", re.M)
MULTI_WS = re.compile(r"\s+")

ENG_RE   = re.compile(r"[A-Za-z]")            # any Latin letter

def clean_pred(text: str) -> str:
    """Minimal cleaning: keep first line that has Latin letters."""
    # normalise escaped newlines
    text = text.replace("\\n", "\n")

    # strip obvious junk
    text = TAG_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MD_RE.sub("", text)

    # examine line-by-line
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # pick first line that contains English / pinyin
        if ENG_RE.search(ln):
            return MULTI_WS.sub(" ", ln)
    # fallback: first non-empty line, even if it’s Chinese only
    for ln in text.splitlines():
        if ln.strip():
            return MULTI_WS.sub(" ", ln.strip())
    return ""

def process(inp_path: pathlib.Path, out_path: pathlib.Path):
    with inp_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            rec["pred"] = clean_pred(rec["pred"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 5b_clean_preds.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    process(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
