### Baseline (no adapter)
# python 5_generate.py --test data/proc/test.jsonl --out infer/baseline_pred.jsonl

### Stacked adapters (accuracy + style)
# python 5_generate.py --acc_ckpt   checkpoints/mistral_lora/final_adapter --style_ckpt checkpoints/mistral_style/final_adapter_style --out infer/stacked_pred.jsonl

"""
Jennifer Xu (Jennifer.Xu.26@dartmouth.edu)

This code takes Chinese poem lines as input and applies both lexical and style LoRA adapters
during inference with the stacked model.
It loads the Mistral-7B model, merges any provided adapters, determines the appropriate prompt,
and batch-generates English translations.
The generated English translations are outputted as a JSONL file infer/test_predictions.jsonl.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, json, pathlib, re
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
import torch

# constants & defaults
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TEST_DEF   = Path("data/proc/test.jsonl")
OUT_DEF    = Path("infer/test_pred.jsonl")
TEMP_DEF   = 0.8
TOPP_DEF   = 0.9
BATCH_DEF  = 4

# CLI
ap = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Generate translations with (optionally) stacked LoRA adapters.",
)
ap.add_argument("--acc_ckpt",   type=Path, default=None,
                help="Path to lexical‑accuracy LoRA adapter (optional)")
ap.add_argument("--style_ckpt", type=Path, default=None,
                help="Path to writing‑style LoRA adapter (optional)")
ap.add_argument("--test", type=Path, default=TEST_DEF, help="Test jsonl file")
ap.add_argument("--out",  type=Path, default=OUT_DEF,  help="Output jsonl file")
ap.add_argument("--temp", type=float, default=TEMP_DEF, help="Sampling temperature")
ap.add_argument("--p",    type=float, default=TOPP_DEF, help="top‑p nucleus sampling")
ap.add_argument("--batch",type=int,   default=BATCH_DEF, help="Batch size (# prompts)")

args = ap.parse_args()

# model & tokenizer
print("Loading base model…")
quant_cfg = BitsAndBytesConfig(llm_int8_threshold=6.0)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=quant_cfg,
)

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.eos_token_id

# load one LoRA and optionally merge
def load_and_merge(adapter_path: Path):
    if adapter_path is None:  # no adapter
        return
    print(f"→  merging adapter {adapter_path}")
    model_peft = PeftModel.from_pretrained(model, adapter_path, device_map="auto")
    model_peft = model_peft.merge_and_unload()
    return model_peft

# merge adapters (accuracy + style)
if args.acc_ckpt:
    model = load_and_merge(args.acc_ckpt)  # accuracy adapter
if args.style_ckpt:
    model = load_and_merge(args.style_ckpt)  # style adapter

model.eval()
print("Model ready. 8‑bit weights: ", any(p.dtype == torch.int8 for p in model.parameters()))

# stop generation exactly at </trans>
class StopOnTransEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        self.end_ids = tokenizer.encode("</trans>")

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -len(self.end_ids):].tolist() == self.end_ids

stopper = StoppingCriteriaList([StopOnTransEnd(tok)])

# data
print("Loading test set →", args.test)

test_ds = load_dataset("json", data_files=str(args.test), split="train")
args.out.parent.mkdir(parents=True, exist_ok=True)

out_f = args.out.open("w", encoding="utf-8")

# generation helpers
BOS = "<cn>"
EOS = "</cn>"

batch_prompts, batch_meta = [], []

def flush():
    if not batch_prompts:
        return
    tokenized = tok(batch_prompts, return_tensors="pt",
                    padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **tokenized,
            max_new_tokens=128,
            do_sample=True,
            temperature=args.temp,
            top_p=args.p,
            pad_token_id=tok.eos_token_id,
            stopping_criteria=stopper,
        )
    for meta, ids in zip(batch_meta, gen):
        full = tok.decode(ids, skip_special_tokens=True)

        # clean up
        # 1) take text after first <trans>
        sek = full.split("<trans>", 1)[-1]
        # 2) stop at first closing tag or new tag
        pred = re.split(r"</trans>|<cn>|</", sek, 1)[0].strip()

        meta["pred"] = pred
        out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    batch_prompts.clear(); batch_meta.clear()

for rec in tqdm(test_ds, desc="generate"):
    prompt = f"<cn>{rec['zh']}</cn>\n" # Chinese source
    prompt += "<trans>" # request translation

    batch_prompts.append(prompt)
    batch_meta.append(dict(rec))
    if len(batch_prompts) >= args.batch:
        flush()
flush()

print("Translations written to", args.out.resolve())
