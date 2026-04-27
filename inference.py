"""
Run the fine-tuned CivBot adapter against the Mistral-7B base model.

Two modes:
  --leader / --state    one-shot generation, prints a single line
  (no flags)            interactive REPL: prompts for leader and state each turn

Examples:
  python inference.py --leader Cleopatra --state "Greeting, Hostile"
  python inference.py --adapter civbot-lora/final
  python inference.py --merged   # if you've merged the adapter into the base
"""

import argparse

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "civbot-lora/final"
INSTRUCTION = (
    "Generate dynamic dialogue for the following Civilization VI leader "
    "based on the game state."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=DEFAULT_MODEL,
                   help="Base model id or local path.")
    p.add_argument("--adapter", default=DEFAULT_ADAPTER,
                   help="Path to the trained LoRA adapter directory.")
    p.add_argument("--merged", action="store_true",
                   help="Skip adapter loading; --model_name already contains merged weights.")
    p.add_argument("--leader", default=None, help="Leader name (one-shot mode).")
    p.add_argument("--state", default=None, help="Game state description (one-shot mode).")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--no_4bit", action="store_true",
                   help="Load in full precision instead of 4-bit (needs more VRAM).")
    return p.parse_args()


def build_prompt(leader: str, state: str) -> str:
    user_input = f"Leader: {leader.strip()}\nState: {state.strip()}"
    user_msg = f"{INSTRUCTION}\n\n{user_input}"
    return f"<s>[INST] {user_msg} [/INST]"


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if not args.no_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)

    if not args.merged:
        print(f"Attaching LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, leader: str, state: str, args) -> str:
    prompt = build_prompt(leader, state)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def repl(model, tokenizer, args) -> None:
    print("\nCivBot interactive mode. Ctrl+C to exit.\n")
    while True:
        try:
            leader = input("Leader: ").strip()
            if not leader:
                continue
            state = input("State : ").strip()
            if not state:
                continue
            line = generate(model, tokenizer, leader, state, args)
            print(f"\n{leader}: {line}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args)

    if args.leader and args.state:
        line = generate(model, tokenizer, args.leader, args.state, args)
        print(f"{args.leader}: {line}")
    else:
        repl(model, tokenizer, args)


if __name__ == "__main__":
    main()
