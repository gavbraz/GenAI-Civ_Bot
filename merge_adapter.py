"""
Optional: merge the LoRA adapter into the base model and export full weights.

Use this if you want a single self-contained model directory Requires enough
RAM/VRAM to hold the un-quantised base model.

    python merge_adapter.py --adapter civbot-lora/final --output civbot-merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter", default="civbot-lora/final")
    p.add_argument("--output", default="civbot-merged")
    args = p.parse_args()

    print(f"Loading base in fp16: {args.model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    print(f"Attaching adapter: {args.adapter}")
    merged = PeftModel.from_pretrained(base, args.adapter)
    print("Merging weights...")
    merged = merged.merge_and_unload()

    print(f"Saving merged model to: {args.output}")
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
