"""Recover the SFT LoRA adapter from an FSDP2 distcp checkpoint and produce:
  <out>/adapter/   standard PEFT adapter (safetensors)
  <out>/merged/    base model with adapter merged (bf16), servable by vLLM

Needed because trainer.save_model() under FSDP2 SHARDED_STATE_DICT wrote only
the model card; the trained weights live in checkpoint-*/pytorch_model_fsdp_0.

Usage:
  python recover_sft_adapter.py \
      --dcp .../checkpoint-354/pytorch_model_fsdp_0 \
      --base Qwen/Qwen2.5-Coder-7B-Instruct \
      --out /eagle/argonne_tpc/lyb/models/qwen7b_sft_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tempfile

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def load_dcp_flat(dcp_dir: str) -> dict:
    """Load a distcp checkpoint into a plain in-memory state dict (single process)."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    dcp_to_torch_save(dcp_dir, tmp_path)
    sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
    Path(tmp_path).unlink(missing_ok=True)
    # dcp_to_torch_save may nest under 'model' or return the flat dict directly
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    args = ap.parse_args()

    print("loading distcp shards ...", flush=True)
    flat = load_dcp_flat(args.dcp)
    # trainer wraps the peft model once more: strip the leading "model."
    renamed = {}
    for k, v in flat.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        renamed[nk] = v.to(torch.bfloat16) if torch.is_tensor(v) else v
    lora_keys = [k for k in renamed if "lora" in k]
    print(f"{len(lora_keys)} lora tensors (sample: {lora_keys[0]})", flush=True)

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading base model on CPU ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    peft_config = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, peft_config)
    result = set_peft_model_state_dict(model, renamed)
    unexpected = getattr(result, "unexpected_keys", [])
    print(f"unexpected keys: {len(unexpected)}", flush=True)
    loaded = sum("lora" in k for k in renamed)
    assert loaded == len(lora_keys)

    out = Path(args.out)
    print("saving adapter ...", flush=True)
    model.save_pretrained(out / "adapter")

    print("merging + saving full model (bf16, ~15GB) ...", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(out / "merged")
    print(f"DONE: {out}/adapter and {out}/merged", flush=True)


if __name__ == "__main__":
    main()
