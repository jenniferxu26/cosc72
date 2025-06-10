### python 5b_clean_preds.py infer/baseline_pred.jsonl infer/baseline_pred_clean.jsonl
### python 5b_clean_preds.py infer/stacked_pred.jsonl  infer/stacked_pred_clean.jsonl

"""
Jennifer Xu (Jennifer.Xu.26@dartmouth.edu)

This code takes the generated translations as input.
It cleans the translations by removing HTML tags, extra whitespace, and any surrounding markup.
The cleaned output is saved as a new JSONL file with the suffix _clean.jsonl.
"""

import json, re, sys, pathlib

TAG_RE   = re.compile(r"</?[^>]+?>")    # strip XML/HTML tags
URL_RE   = re.compile(r"https?://\S+")  # strip bare URLs
MULTI_WS = re.compile(r"\s+")
ENG_RE   = re.compile(r"[A-Za-z]")      # any Latin letter

def clean_pred(text: str) -> str:
    # keep the first block of lines that contain letters
    # stop when hit blank line or line starts with "<"

    # normalise \n \\n
    text = text.replace("\\n", "\n")

    # filters
    text = TAG_RE.sub("", text)
    text = URL_RE.sub("", text)

    # split and trim
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # locate the first “real” English/pinyin line
    start = None
    for i, ln in enumerate(lines):
        if ENG_RE.search(ln) and not ln.startswith("<"):
            start = i
            break
    if start is None:                       # fallback – Chinese only
        return MULTI_WS.sub(" ", lines[0]) if lines else ""

    # collect lines until we hit markup
    kept = []
    for ln in lines[start:]:
        if ln.startswith("<") or not ENG_RE.search(ln):
            break
        kept.append(ln)

    return MULTI_WS.sub(" ", " ".join(kept))

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
