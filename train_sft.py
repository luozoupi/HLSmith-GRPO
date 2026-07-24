"""SFT entrypoint (TRL SFTTrainer).

Launch (4 GPUs, FSDP):
  accelerate launch --config_file configs/fsdp.yaml train_sft.py \
      --model Qwen/Qwen2.5-Coder-7B-Instruct --dataset <path> --output_dir <dir>

Dataset: a load_from_disk dir or a .jsonl with either a "messages" column
(chat format) or a "text" column — TRL handles both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset, load_from_disk
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


def load_any(path: str):
    p = Path(path)
    if p.is_dir():
        return load_from_disk(str(p))
    if p.suffix in (".json", ".jsonl"):
        return load_dataset("json", data_files=str(p), split="train")
    raise ValueError(f"unsupported dataset path: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--lora", action="store_true",
                    help="LoRA instead of full fine-tune (full 7B FT OOMs on 4x A100-40G)")
    ap.add_argument("--lora-r", type=int, default=32, help="LoRA rank (scale capacity: 32/64/128)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="HF gradient checkpointing (use with an FSDP config that has "
                         "fsdp_activation_checkpointing OFF; needed for long context — FSDP2's "
                         "own activation checkpointing does not reduce the activation spike here)")
    ap.add_argument("--liger", action="store_true",
                    help="Liger fused kernels (fused linear-CE + RMSNorm/SwiGLU): large activation "
                         "+ logit memory reduction — the enabler for big models at long context")
    ap.add_argument("--qlora", action="store_true",
                    help="4-bit QLoRA base (bitsandbytes nf4 + bf16 compute); implies --lora. "
                         "Frees ~4x param memory for the biggest models. Use a non-FSDP config "
                         "(configs/ddp.yaml) — FSDP+bnb is fragile.")
    args = ap.parse_args()
    if args.qlora:
        args.lora = True

    peft_config = None
    # Base load dtype: SFTConfig(bf16=True) only enables mixed-precision *training* —
    # from_pretrained still loads weights in fp32 by default (~2x memory). For LoRA the
    # base is frozen, so loading it directly in bf16 is standard and halves param memory
    # (the fix that lets big models fit long context on 40GB A100s). Full-FT path unchanged.
    model_init_kwargs = {"attn_implementation": "sdpa"}
    if args.lora:
        import torch
        from peft import LoraConfig

        model_init_kwargs["dtype"] = torch.bfloat16
        if args.qlora:
            from transformers import BitsAndBytesConfig
            model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                # bf16 storage lets FSDP2 SHARD the 4-bit weights (else each GPU holds the
                # full model → OOM). Required for FSDP+QLoRA (Answer.AI recipe).
                bnb_4bit_quant_storage=torch.bfloat16)
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_len,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=10,
        bf16=True,
        # --grad-ckpt enables HF gradient checkpointing (reliable with PEFT). Pair it with
        # an FSDP config that has fsdp_activation_checkpointing OFF (configs/fsdp_gc.yaml) to
        # avoid double-wrapping. Default False preserves prior behavior (relies on FSDP's).
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
        use_liger_kernel=args.liger,
        report_to=["tensorboard"],
        model_init_kwargs=model_init_kwargs,
    )
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=load_any(args.dataset),
                         peft_config=peft_config)

    # One-shot memory/param probe (XKERNEL_MEM_PROBE=1): prints per-rank GPU memory,
    # total/trainable params and dtype right after accelerate prepares the model
    # (post-FSDP-wrap, pre first forward), then exits — to diagnose OOM sources.
    import os as _os
    if _os.environ.get("XKERNEL_MEM_PROBE"):
        import torch as _torch
        from transformers import TrainerCallback

        class _MemProbe(TrainerCallback):
            def on_train_begin(self, a, s, c, model=None, **kw):
                dev = _torch.cuda.current_device()
                alloc = _torch.cuda.memory_allocated(dev) / 1e9
                resv = _torch.cuda.memory_reserved(dev) / 1e9
                tot = sum(p.numel() for p in model.parameters())
                tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
                dts = {str(p.dtype) for p in model.parameters()}
                print(f"[MEMPROBE dev{dev}] alloc={alloc:.2f}GB reserved={resv:.2f}GB "
                      f"total_params={tot/1e9:.3f}B trainable={tr/1e6:.1f}M dtypes={dts}",
                      flush=True)
                raise SystemExit(0)
        trainer.add_callback(_MemProbe())

    last = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(args.output_dir + "/final")


if __name__ == "__main__":
    main()
