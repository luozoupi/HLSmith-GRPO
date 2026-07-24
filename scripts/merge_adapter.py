"""Merge a PEFT LoRA adapter into its base model and save (bf16, safetensors).

Usage: python merge_adapter.py --base <hf-id> --adapter <dir> --out <dir>
Skips silently if <out>/config.json already exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "config.json").exists():
        print(f"merged model already at {out}; skipping")
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    print(f"MERGE DONE -> {out}")


if __name__ == "__main__":
    main()
