"""
QLoRA fine-tuning of Mistral-7B-Instruct on the Civ VI leader-dialogue dataset.

The dataset (civ 6_finetune_dataset.jsonl) is Alpaca-style with fields:
    instruction : task description
    input       : "Leader: <name>\nState: <state>"
    output      : the in-character line the leader should say

format each example into Mistral's instruction template, then train LoRA
adapters on a 4-bit-quantised base model so it fits on a single consumer GPU.
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_DATA = "civ6_finetune_dataset.jsonl"
DEFAULT_OUTPUT = "civbot-lora"


@dataclass
class Args:
    model_name: str
    data_path: str
    output_dir: str
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    max_length: int
    eval_split: float
    seed: int


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--eval_split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Args(**vars(a))


def format_example(ex: dict) -> str:
    """Render an Alpaca record in Mistral's [INST] template."""
    instruction = ex["instruction"].strip()
    user_input = ex.get("input", "").strip()
    user_msg = f"{instruction}\n\n{user_input}" if user_input else instruction
    return f"<s>[INST] {user_msg} [/INST] {ex['output'].strip()}</s>"


def load_dataset(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading dataset: {args.data_path}")
    raw = load_dataset(args.data_path)
    raw = raw.shuffle(seed=args.seed)

    def tokenize(batch):
        texts = [format_example(
            {"instruction": ins, "input": inp, "output": out}
        ) for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"])]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        return enc

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )

    if args.eval_split > 0:
        split = tokenized.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = tokenized, None

    print(f"Train examples: {len(train_ds)}; eval: {len(eval_ds) if eval_ds else 0}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model in 4-bit: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    print(f"Saving LoRA adapter + tokenizer to: {final_dir}")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done.")


if __name__ == "__main__":
    main()
