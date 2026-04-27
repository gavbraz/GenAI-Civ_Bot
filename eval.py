"""
Evaluate the fine-tuned CivBot model on a held-out slice of the dataset.

Default backend is a locally-running Ollama instance 

Metrics:
    - corpus BLEU-4 (sacrebleu)
    - mean character-level similarity (difflib SequenceMatcher ratio)
    - per-leader BLEU breakdown
A few side-by-side prediction/reference pairs are printed at the end.

Usage:
    pip install requests sacrebleu
    python eval.py --num_samples 50
    python eval.py --backend hf --adapter civbot-lora/final
"""

import argparse
import json
import random
from collections import defaultdict
from difflib import SequenceMatcher

import requests
import sacrebleu


INSTRUCTION = (
    "Generate dynamic dialogue for the following Civilization VI leader "
    "based on the game state."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="civ6_finetune_dataset.jsonl")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["ollama", "hf"], default="ollama")
    p.add_argument("--ollama_url", default="http://localhost:11434/api/generate")
    p.add_argument("--ollama_model", default="civbot")
    p.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter", default="civbot-lora/final")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--show", type=int, default=5, help="How many examples to print.")
    return p.parse_args()


def load_holdout(path, num, seed):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:num]


def build_user_message(ex):
    return f"{INSTRUCTION}\n\n{ex['input'].strip()}"


def gen_ollama(args, user_msg):
    r = requests.post(args.ollama_url, json={
        "model": args.ollama_model,
        "prompt": user_msg,
        "stream": False,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": args.max_new_tokens,
        },
    }, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def make_hf_generator(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    @torch.inference_mode()
    def gen(user_msg):
        prompt = f"<s>[INST] {user_msg} [/INST]"
        inp = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inp,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
        return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return gen


def leader_of(ex):
    for line in ex["input"].splitlines():
        if line.lower().startswith("leader:"):
            return line.split(":", 1)[1].strip()
    return "?"


def main():
    args = parse_args()
    examples = load_holdout(args.data_path, args.num_samples, args.seed)

    if args.backend == "ollama":
        gen = lambda msg: gen_ollama(args, msg)
    else:
        gen = make_hf_generator(args)

    preds, refs = [], []
    by_leader = defaultdict(lambda: {"preds": [], "refs": []})

    print(f"Generating {len(examples)} predictions via {args.backend}...")
    for i, ex in enumerate(examples, 1):
        user_msg = build_user_message(ex)
        try:
            pred = gen(user_msg)
        except Exception as e:
            print(f"  [{i}] generation failed: {e}")
            continue
        ref = ex["output"].strip()
        preds.append(pred)
        refs.append(ref)
        leader = leader_of(ex)
        by_leader[leader]["preds"].append(pred)
        by_leader[leader]["refs"].append(ref)
        if i % 10 == 0:
            print(f"  {i}/{len(examples)}")

    if not preds:
        print("No successful generations. Is Ollama running and the model created?")
        return

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    char_sim = sum(SequenceMatcher(None, p, r).ratio() for p, r in zip(preds, refs)) / len(preds)

    print("\n=== Aggregate ===")
    print(f"  Samples       : {len(preds)}")
    print(f"  BLEU-4        : {bleu:.2f}")
    print(f"  Char similarity: {char_sim:.3f}")

    print("\n=== Per-leader BLEU ===")
    for leader, data in sorted(by_leader.items()):
        if len(data["preds"]) < 2:
            continue
        b = sacrebleu.corpus_bleu(data["preds"], [data["refs"]]).score
        print(f"  {leader:<14} n={len(data['preds']):<3} BLEU={b:.2f}")

    print("\n=== Examples ===")
    for ex, pred, ref in zip(examples[:args.show], preds[:args.show], refs[:args.show]):
        print(f"\n  [{leader_of(ex)}] {ex['input'].splitlines()[-1]}")
        print(f"    REF : {ref}")
        print(f"    PRED: {pred}")


if __name__ == "__main__":
    main()
