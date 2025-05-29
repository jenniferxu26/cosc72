OPENAI_KEY = "sk-XXX"

import argparse, json, os, random, re, sys, pathlib
from statistics import mean

# default paths
REFS_DEF = pathlib.Path("data/proc/test.jsonl")
BASE_DEF = pathlib.Path("infer/test_pred_base.jsonl")
LORA_DEF = pathlib.Path("infer/test_pred_lora.jsonl")

# CLI
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--refs", default=REFS_DEF, type=pathlib.Path,
                   help="reference JSONL (has field 'en')")
    p.add_argument("--base", default=BASE_DEF, type=pathlib.Path,
                   help="baseline JSONL (has field 'pred')")
    p.add_argument("--lora", default=LORA_DEF, type=pathlib.Path,
                   help="LoRA JSONL (has field 'pred')")
    p.add_argument("--sample", type=int, default=30,
                   help="GPT-4 sample size per system")
    return p.parse_args()

args = cli()
for fp in (args.refs, args.base, args.lora):
    if not fp.exists():
        sys.exit(f"❌  File not found: {fp}")

# load JSONL & align
def load_jsonl(path, field):
    out = []
    with path.open(encoding="utf-8") as f:
        for ln in f:
            try:
                out.append(json.loads(ln)[field])
            except Exception:
                print(f"⚠ skipped malformed line in {path}")
    return out

refs  = load_jsonl(args.refs, "en")
base  = load_jsonl(args.base, "pred")
lora  = load_jsonl(args.lora, "pred")

# BLEU & METEOR
import sacrebleu, nltk
from nltk.translate.meteor_score import meteor_score
nltk.download("punkt", quiet=True); nltk.download("wordnet", quiet=True)

def corpus_meteor(ref, hyp):
    rt = [r.split() for r in ref]
    ht = [h.split() for h in hyp]
    return mean(meteor_score([r], h) for r, h in zip(rt, ht))*100

metrics = {}
for tag, hyp in (("baseline", base), ("LoRA", lora)):
    metrics[tag] = {
        "BLEU"  : sacrebleu.corpus_bleu(hyp, [refs]).score,
        "METEOR": corpus_meteor(refs, hyp),
    }

# GPT-4 Tri-Score
import openai
if OPENAI_KEY.startswith("sk-XXX") or not OPENAI_KEY:
    print("⚠️  GPT-4 key missing → Tri-Score skipped")
else:
    openai.api_key = OPENAI_KEY
    def tri(ref, hyp, n):
        idx  = random.sample(range(len(ref)), n)
        scores = []
        for i in idx:
            prompt = f"""You are an expert bilingual evaluator.

REF: {ref[i]}

MT : {hyp[i]}

Score 0-10 for adequacy, fluency, poetic style.
Return JSON: {{"adeq":x,"flue":y,"style":z}}"""
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
        return mean(scores) if scores else None

    metrics["baseline"]["Tri"] = tri(refs, base, args.sample)
    metrics["LoRA"]["Tri"]     = tri(refs, lora, args.sample)

# output
def row(tag, m):
    tri = f"{m.get('Tri', 0):4.2f}" if m.get("Tri") is not None else "—"
    return f"{tag:<8} | {m['BLEU']:6.2f} | {m['METEOR']:6.2f} | {tri:>5}"

print("Model | BLEU | METEOR | Tri")
print(row("baseline", metrics["baseline"]))
print(row("LoRA ", metrics["LoRA"]))

# save as csv
import csv, pathlib
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