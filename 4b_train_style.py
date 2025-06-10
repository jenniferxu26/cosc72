### python 4b_train_style.py --hf_token hf_XXX

"""
Jennifer Xu (Jennifer.Xu.26@dartmouth.edu)

This code loads the Gutenberg English corpus.
It tokenizes each line under a style prompt, loads Mistral-7B in 4-bit with the lexical adapter merged,
adds a new LoRA “style” adapter on the attention projections, and fine-tunes that adapter for one epoch.
The fine-tuned style adapter weights and tokenizer are saved to checkpoints/mistral_style/final_adapter_style/,
ready to be used during translation generation.
"""

import os, math, time, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
    __version__ as HF_VER,
)
from packaging import version
from peft import LoraConfig, PeftModel

# paths
CSV_PATH      = Path("data/raw/gutenberg_en.csv")
BASE_ADAPTER  = Path("checkpoints/mistral_lora/final_adapter")
OUT_DIR       = Path("checkpoints/mistral_style")
MODEL_NAME    = "mistralai/Mistral-7B-Instruct-v0.3"

EPOCHS, LR, BATCH = 1, 5e-5, 2
RANK, ALPHA, SEED = 8, 16, 42

# helper function
def build_example(rec, tok):
    prompt_ids  = tok("<style>", add_special_tokens=False)["input_ids"]
    answer_ids  = tok(rec["text"].strip() + tok.eos_token, add_special_tokens=False)["input_ids"]
    rec["input_ids"] = prompt_ids + answer_ids
    rec["labels"]    = [-100]*len(prompt_ids) + answer_ids
    return rec

def main():
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()
    if not args.hf_token:
        parser.error("Supply --hf_token or set $HF_TOKEN")

    login(token=args.hf_token, add_to_git_credential=False)
    print("Hugging Face authenticated.")

    # dataset
    ds_style = load_dataset("csv", data_files=str(CSV_PATH), split="train")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True); tok.pad_token = tok.eos_token
    ds_proc = ds_style.map(lambda r: build_example(r, tok), remove_columns=ds_style.column_names)
    train_ds, val_ds = ds_proc.train_test_split(test_size=0.05, seed=SEED).values()

    # model with base adapter
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, BASE_ADAPTER)

    # add style adapter
    lcfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model.add_adapter("style", lcfg); model.set_adapter("style")

    train_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=max(1, 8 // BATCH),
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=1,
        seed=SEED,
        report_to="none",
        eval_strategy="no",
        save_strategy="no",
    )

    collator = DataCollatorForSeq2Seq(tok, tok, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)

    t0 = time.time(); trainer.train()
    print("Training finished.")

    out = OUT_DIR/"final_adapter_style"; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    print("Style adapter saved", out)

if __name__ == "__main__":
    main()
