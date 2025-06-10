### python 4_train_model.py --hf_token hf_XXX

"""
Jennifer Xu (Jennifer.Xu.26@dartmouth.edu)

This code loads the glossed training dataset and retrieves GPT-4o explanations from explanations.sqlite.
It logs into Hugging Face, builds the prompts, tokenizes them, loads the base model,
wraps it with the LoRA adapter, and fine-tunes for 2 epochs.
The output is saved to checkpoints/mistral_lora/final_adapter, where all model weights and
the tokenizer are stored, ready to be used for translation generation.
"""

import os, math, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math, time, argparse
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# path
TRAIN_JSONL = Path("data/proc/train_gloss.jsonl")
OUT_DIR     = Path("checkpoints/mistral_lora")
MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.3"

# parameter
BATCH    = 2
EPOCHS   = 2
LR       = 1e-4
RANK     = 8
ALPHA    = 16
VAL_FRAC = 0.05
SEED     = 42

# helper functions
def build_example(rec, tok, ctx_lookup=None):
    # pull the extra explanation, if available
    context = ""
    if ctx_lookup is not None:
        context = ctx_lookup.get(rec["zh"], "").strip()

    # build the prompt template
    prompt = (
            f"<cn>{rec['zh']}</cn>\n"
            f"<gloss>{rec['gloss']}</gloss>\n"
            + (f"<context>{context}</context>\n" if context else "")
            + "<trans>"
    )

    # tokenize prompt + answer exactly as before
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(rec["en"] + tok.eos_token, add_special_tokens=False)["input_ids"]

    rec["input_ids"] = prompt_ids + answer_ids
    rec["labels"] = [-100] * len(prompt_ids) + answer_ids
    return rec

def main():
    # CLI
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face access token (or set $HF_TOKEN)",
    )
    args = ap.parse_args()

    # Hugging Face login
    login(token=args.hf_token, add_to_git_credential=False)
    print("Hugging Face authenticated.")

    # load data
    ds_raw = load_dataset("json", data_files=str(TRAIN_JSONL), split="train")
    print(f"Training data: {len(ds_raw)} rows.")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.pad_token = tok.eos_token

    import sqlite3
    ctx = {}
    with sqlite3.connect("data/proc/explanations.sqlite") as conn:
        for zh, expl in conn.execute(
                "SELECT zh, explanation FROM explain"):  # <- change here
            ctx[zh] = expl

    ds_proc = ds_raw.map(lambda r: build_example(r, tok, ctx), remove_columns=ds_raw.column_names)
    train_ds, val_ds = ds_proc.train_test_split(test_size=VAL_FRAC, seed=SEED).values()
    print("Train/test split:", len(train_ds), "/", len(val_ds))

    # model & LoRA
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    print("Model loaded.")

    # LoRA
    lora_cfg = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # training
    steps = math.ceil(len(train_ds) / BATCH) * EPOCHS
    print(f"Optimiser steps = {steps:,}")

    train_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=max(1, 8 // BATCH),
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=max(50, steps // 10),
        save_steps=max(50, steps // 10),
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(tok, tok, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    # train
    tic = time.time()
    trainer.train()
    print("Training finished.")

    # save adapter & test
    save_dir = OUT_DIR / "final_adapter"
    model.save_pretrained(save_dir);
    tok.save_pretrained(save_dir)
    print("LoRA adapter saved: ", save_dir)

    sample = train_ds.shuffle(seed=0)[0]
    prompt = tok.decode(sample["input_ids"], skip_special_tokens=True)
    gen = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                         max_new_tokens=60, do_sample=False)

if __name__ == "__main__":
    main()
