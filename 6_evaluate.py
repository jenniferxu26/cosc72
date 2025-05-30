### python 6_evaluate.py --hf_key "hf_XXX"

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, json, random, re, sys, pathlib, csv
from statistics import mean
import sacrebleu, nltk
from nltk.translate.meteor_score import meteor_score
nltk.download("punkt", quiet=True); nltk.download("wordnet", quiet=True)

# CLI
DEF_REFS = pathlib.Path("data/proc/test.jsonl")
DEF_BASE = pathlib.Path("infer/baseline_pred.jsonl")
DEF_LORA = pathlib.Path("infer/stacked_pred.jsonl")

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--refs",  type=pathlib.Path, default=DEF_REFS)
ap.add_argument("--base",  type=pathlib.Path, default=DEF_BASE)
ap.add_argument("--lora",  type=pathlib.Path, default=DEF_LORA)
ap.add_argument("--sample",type=int, default=30, help="GPT‑4 sample size per system")
ap.add_argument("--hf_key",type=str, default=os.getenv("OPENAI_KEY"), help="OpenAI API key or set $OPENAI_KEY")
args = ap.parse_args()

# helper functions
def load_jsonl(path, field):
    out = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            try:
                out.append(json.loads(ln)[field])
            except Exception:
                print(f"Skipped malformed line in {path}")
    return out

def corpus_meteor(ref, hyp):
    rt = [r.split() for r in ref]; ht = [h.split() for h in hyp]
    return mean(meteor_score([r], h) for r, h in zip(rt, ht))*100

# load data
refs  = load_jsonl(args.refs, "en")
base  = load_jsonl(args.base, "pred")
lora  = load_jsonl(args.lora, "pred")

metrics = {}
for tag, hyp in (("baseline", base), ("LoRA", lora)):
    metrics[tag] = {
        "BLEU"  : sacrebleu.corpus_bleu(hyp, [refs]).score,
        "METEOR": corpus_meteor(refs, hyp),
    }

# GPT‑4o Tri‑Score (optional)
import openai, time
openai.api_key = args.hf_key

def tri(ref, hyp, n):
    idx  = random.sample(range(len(ref)), n)
    scores = []
    for i in idx:
        prompt = f"""You are an expert bilingual evaluator.\n\nREF: {ref[i]}\n\nMT : {hyp[i]}\n\nScore 0-10 for adequacy, fluency, poetic style.\nReturn JSON: {{\"adeq\":x,\"flue\":y,\"style\":z}}"""
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,max_tokens=30)
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```")
        m   = re.search(r"\{.*\}", raw, flags=re.S)
        if not m: continue
        try:
            j = json.loads(m.group(0))
            scores.append((j["adeq"]+j["flue"]+j["style"])/3)
        except Exception:
            continue
        time.sleep(0.5)            # be gentle with rate limits
    return mean(scores) if scores else None

metrics["baseline"]["Tri"] = tri(refs, base, args.sample)
metrics["LoRA"]["Tri"]     = tri(refs, lora, args.sample)

# result table
def row(tag, m):
    tri = f"{m.get('Tri', 0):4.2f}" if m.get("Tri") is not None else "—"
    return f"{tag:<8} | {m['BLEU']:6.2f} | {m['METEOR']:6.2f} | {tri:>5}"

print("Model     |  BLEU | METEOR | Tri")
print(row("baseline", metrics["baseline"]))
print(row("LoRA",     metrics["LoRA"]))

# save CSV
csv_path = pathlib.Path("results.csv")
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["system", "BLEU", "METEOR", "Tri"])
    w.writerow(["baseline",
                f"{metrics['baseline']['BLEU']:.2f}",
                f"{metrics['baseline']['METEOR']:.2f}",
                f"{metrics['baseline'].get('Tri',''):.2f}" if metrics['baseline'].get('Tri') else ""])
    w.writerow(["LoRA",
                f"{metrics['LoRA']['BLEU']:.2f}",
                f"{metrics['LoRA']['METEOR']:.2f}",
                f"{metrics['LoRA'].get('Tri',''):.2f}" if metrics['LoRA'].get('Tri') else ""])
