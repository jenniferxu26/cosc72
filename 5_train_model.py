import os, math, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from huggingface_hub import login
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

TRAIN_JSONL = Path("data/proc/train_gloss.jsonl")
OUT_DIR     = Path("checkpoints/mistral_lora")
MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.3"

BATCH   = 2
EPOCHS  = 2
LR      = 1e-4
RANK    = 8
ALPHA   = 16
VAL_FRAC = 0.05

# Hugging Face setup
HF_TOKEN_HARDCODED = "hf_***************************"
login(token=HF_TOKEN_HARDCODED, add_to_git_credential=False)
print("Hugging Face authenticated.")

# load data
ds_raw = load_dataset("json", data_files=str(TRAIN_JSONL), split="train")
print(f"Training data: {len(ds_raw)} rows")

# build prompt
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.pad_token = tok.eos_token

def build_example(rec):
    # encode prompt/answer separately
    prompt_ids = tok(f"<cn>{rec['zh']}</cn>\n<gloss>{rec['gloss']}</gloss>\n<trans>",
                     add_special_tokens=False)["input_ids"]
    answer_ids = tok(rec["en"] + tok.eos_token, add_special_tokens=False)["input_ids"]

    # input == labels in length
    rec["input_ids"] = prompt_ids + answer_ids

    # mask prompt tokens in labels with -100
    rec["labels"] = [-100] * len(prompt_ids) + answer_ids
    return rec

ds_proc = ds_raw.map(build_example, remove_columns=ds_raw.column_names)

train_ds, val_ds = ds_proc.train_test_split(test_size=VAL_FRAC, seed=42).values()
print("Training/testing split:", len(train_ds), len(val_ds))

# load model
bnb = BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True)

print("Loading base model ====")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             quantization_config=bnb,
                                             device_map="auto")

# LoRA wrapper
lora_cfg = LoraConfig(
    r=RANK, lora_alpha=ALPHA, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# setup trainer
steps = math.ceil(len(train_ds)/BATCH)*EPOCHS
print(f"Optimizer steps = {steps:,}")

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
print(f"Training completed.")

# save adapter & test
save_dir = OUT_DIR / "final_adapter"
model.save_pretrained(save_dir); tok.save_pretrained(save_dir)
print("LoRA adapter saved: ", save_dir)

sample = train_ds.shuffle(seed=0)[0]
prompt = tok.decode(sample["input_ids"], skip_special_tokens=True)
gen = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                     max_new_tokens=60, do_sample=False)

## test
print("Sample translation: \n",
      tok.decode(gen[0], skip_special_tokens=True).split("<trans>")[-1])