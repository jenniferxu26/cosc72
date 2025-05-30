"""
# Baseline (no adapters)
python 5_generate.py --test data/proc/test.jsonl \
                    --out  infer/baseline_pred.jsonl

# Stacked adapters (accuracy + style)
python 5_generate.py \
        --acc_ckpt   checkpoints/mistral_lora/final_adapter \
        --style_ckpt checkpoints/mistral_style/final_adapter_style \
        --out        infer/stacked_pred.jsonl
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, json, pathlib, torch
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# paths & defaults
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TEST_DEF   = Path("data/proc/test.jsonl")
OUT_DEF    = Path("infer/test_pred.jsonl")
TEMP_DEF   = 0.8
TOPP_DEF   = 0.9
BATCH_DEF  = 4

# CLI
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--acc_ckpt",  type=Path, default=None, help="Path to accuracy LoRA adapter (translation). If omitted, run baseline model.")
ap.add_argument("--style_ckpt",type=Path, default=None, help="Path to optional style LoRA adapter (diction). Stacked on top of --acc_ckpt.")
ap.add_argument("--test",      type=Path, default=TEST_DEF, help="JSONL test file")
ap.add_argument("--out",       type=Path, default=OUT_DEF,  help="Output JSONL file")
ap.add_argument("--temp",      type=float, default=TEMP_DEF)
ap.add_argument("--p",         type=float, default=TOPP_DEF)
ap.add_argument("--batch",     type=int,   default=BATCH_DEF)
args = ap.parse_args()

# load tokenizer
tok_source = args.acc_ckpt if args.acc_ckpt else BASE_MODEL
tok = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
if tok.pad_token is None:                 # ensure padding token exists
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

# load model (+ adapters)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto"
)

adapters = []

# accuracy adapter
if args.acc_ckpt:
    model = PeftModel.from_pretrained(model, str(args.acc_ckpt), device_map="auto")
    adapters.append(str(args.acc_ckpt))
    model = model.merge_and_unload()

# style adapter (merged on top of accuracy)
if args.style_ckpt:
    model = PeftModel.from_pretrained(
        model, str(args.style_ckpt), adapter_name="style", device_map="auto"
    )
    adapters.append(str(args.style_ckpt))
    model = model.merge_and_unload()

model.eval()

print("Model ready | adapters:", adapters if adapters else "none (baseline)")

# load test data
test_ds = load_dataset("json", data_files=str(args.test), split="train")
print("Test rows:", len(test_ds))

# generate
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8") as out_f:

    batch_prompts, batch_meta = [], []

    def flush():
        if not batch_prompts:
            return
        tokenized = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **tokenized,
                max_new_tokens=128,
                do_sample=True,
                temperature=args.temp,
                top_p=args.p,
                pad_token_id=tok.eos_token_id,
            )
        for meta, ids in zip(batch_meta, gen):
            full = tok.decode(ids, skip_special_tokens=True)
            meta["pred"] = full.split("<trans>")[-1].strip()
            out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        batch_prompts.clear(); batch_meta.clear()

    for rec in tqdm(test_ds, desc="generate"):
        prompt = f"<cn>{rec['zh']}</cn>\n<gloss>{rec['gloss']}</gloss>\n<trans>"
        batch_prompts.append(prompt)
        batch_meta.append(rec)
        if len(batch_prompts) >= args.batch:
            flush()
    flush()

print("Translations written to", args.out.resolve())
