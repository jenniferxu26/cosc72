import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, json, pathlib
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import PeftModel

# parameters
DEF_CKPT  = pathlib.Path("checkpoints/mistral_lora/final_adapter")
DEF_TEST  = pathlib.Path("data/proc/test.jsonl")
DEF_OUT   = pathlib.Path("infer/test_pred.jsonl")
TEMP_DEF  = 0.8
TOPP_DEF  = 0.9
BATCH_DEF = 4

# CLI
p = argparse.ArgumentParser()
p.add_argument("--ckpt",  type=pathlib.Path, default=DEF_CKPT)
p.add_argument("--test",  type=pathlib.Path, default=DEF_TEST)
p.add_argument("--out",   type=pathlib.Path, default=DEF_OUT)
p.add_argument("--temp",  type=float,        default=TEMP_DEF)
p.add_argument("--p",     type=float,        default=TOPP_DEF)
p.add_argument("--batch", type=int,          default=BATCH_DEF)
args = p.parse_args()


# load base + merge LoRA adapter
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True)

tok   = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                             quantization_config=bnb,
                                             device_map="auto")
model = PeftModel.from_pretrained(model, args.ckpt, device_map="auto")
model = model.merge_and_unload()    # faster inference
model.eval()

# load test set
test_ds = load_dataset("json", data_files=str(args.test), split="train")

# generate
args.out.parent.mkdir(parents=True, exist_ok=True)
out_f = args.out.open("w", encoding="utf-8")

batch_prompts, batch_meta = [], []
def flush():
    if not batch_prompts: return
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
        )
    for meta, ids in zip(batch_meta, gen):
        full = tok.decode(ids, skip_special_tokens=True)
        meta["pred"] = full.split("<trans>")[-1].strip()
        out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    batch_prompts.clear(); batch_meta.clear()

for rec in tqdm(test_ds, desc="translate"):
    prompt = f"<cn>{rec['zh']}</cn>\n<gloss>{rec['gloss']}</gloss>\n<trans>"
    batch_prompts.append(prompt)
    batch_meta.append(rec)
    if len(batch_prompts) >= args.batch: flush()
flush()
out_f.close()