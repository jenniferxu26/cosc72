### python 6_evaluate.py --sk_token sk_XXX

"""
Jennifer Xu (Jennifer.Xu.26@dartmouth.edu)

This code takes JSONL files of the baseline predictions, LoRA-enhanced predictions, and the gold English reference lines.
It computes the BLEU score, METEOR score, and prompts GPT-4o to provide a TRI (Translation Rating Index) score
for qualitative assessment.
Finally, it outputs all metrics to results.csv.
"""

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
ap.add_argument("--sk_token",type=str, default=os.getenv("OPENAI_KEY"), help="OpenAI API key or set $OPENAI_KEY")
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

# load data
refs = load_jsonl(args.refs, "en")
refs_zh = load_jsonl(args.refs, "zh")
base = load_jsonl(args.base, "pred")
lora = load_jsonl(args.lora, "pred")

# metric functions
def corpus_meteor(ref, hyp):
    # tokenized text used for BLEU
    return mean(
        meteor_score([r.split()], h.split())    # use METEOR’s internal stem/syn
        for r, h in zip(ref, hyp)
    ) * 100

metrics = {}
for tag, hyp in (("baseline", base), ("LoRA", lora)):
    metrics[tag] = {
        # SACREBLEU’s built-in tokeniser keeps punctuation rules consistent
        "BLEU": sacrebleu.corpus_bleu(
            hyp, [refs], tokenize="13a", lowercase=True
        ).score,
        "METEOR": corpus_meteor(refs, hyp),
    }

# GPT‑4o Tri‑Score
import openai, time
openai.api_key = args.sk_token

def tri(ref, hyp, n):
    idx = random.sample(range(len(ref)), n)
    scores = []
    for i in idx:
        prompt = f"""
        **Source (Chinese)**
        {ref[i]}
        
        **Candidate English translation**
        {hyp[i]}
        
        You are an expert literary translator of Classical Chinese verse.

        Rate the translation on three **independent** 0 – 10 integer scales  
        (10 = excellent, 5 = mixed, 0 = unacceptable).  
        Consider the entire line or couplet, not isolated words.
        
        1. adequacy      – Does it preserve the full *meaning, imagery, and nuance* of the Chinese?  
                            Key ideas and metaphors kept → 8-10   ✗ major content lost or added → < 5  
        2. poetic_style  – Does it *read like poetry* in English?  
                            Look for rhythm, vivid diction, concise phrasing, and (optionally) gentle archaic flavour;  
                            downgrade literal, prosy, or dictionary-like wording.  
        3. fluency        – Is the English *grammatically correct and coherent*?  
                            Penalize awkward syntax or wrong register, not tasteful archaism.
        
        Reply with **ONLY** one JSON object (no markdown, comments, or line breaks).  
        Example:  
        {{"adequacy": 9, "poetic_style": 8, "fluency": 9}}
        """

        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=30)
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```")
        m   = re.search(r"\{.*\}", raw, flags=re.S)
        if not m: continue

        j = json.loads(m.group(0))
        scores.append((j["adequacy"]+j["poetic_style"]+j["fluency"])/3)
        time.sleep(0.5)
    return mean(scores) if scores else None

metrics["baseline"]["Tri"] = tri(refs_zh, base, args.sample)
metrics["LoRA"]["Tri"]     = tri(refs_zh, lora, args.sample)

# result table
def row(tag, m):
    tri = f"{m.get('Tri', 0):4.2f}" if m.get("Tri") is not None else "—"
    return f"{tag:<8} | {m['BLEU']:6.2f} | {m['METEOR']:6.2f} | {tri:>5}"

print("Model     |  BLEU | METEOR | GPT-4o Tri")
print(row("baseline", metrics["baseline"]))
print(row("LoRA",     metrics["LoRA"]))

# save CSV
csv_path = pathlib.Path("results.csv")
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["system", "BLEU", "METEOR", "TRI"])
    w.writerow(["baseline",
                f"{metrics['baseline']['BLEU']:.2f}",
                f"{metrics['baseline']['METEOR']:.2f}",
                f"{metrics['baseline'].get('Tri',''):.2f}" if metrics['baseline'].get('Tri') else ""])
    w.writerow(["LoRA",
                f"{metrics['LoRA']['BLEU']:.2f}",
                f"{metrics['LoRA']['METEOR']:.2f}",
                f"{metrics['LoRA'].get('Tri',''):.2f}" if metrics['LoRA'].get('Tri') else ""])
