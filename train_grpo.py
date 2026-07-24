"""GRPO entrypoint: TRL colocated vLLM rollouts + Vitis HLS QoR rewards.

Launch (4 GPUs):
  accelerate launch --num_processes 4 train_grpo.py \
      --model <hf-id-or-checkpoint> --output_dir <dir> [--lora]

Requires BOTH env scripts sourced (env_llmhls.sh for python, env_vitis.sh for
vitis_hls) and per-task baselines present (python -m hlsenv baseline <task_dir>).
HLS worker knobs come from HLSENV_WORKERS / HLSENV_CORES / HLSENV_SCRATCH / HLSENV_ARCHIVE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from torch.distributed.elastic.multiprocessing.errors import record
from trl import GRPOConfig, GRPOTrainer

from hlsenv.reward import HLSRewardFunc
from hlsenv.tasks import load_tasks

PROJECT = Path(__file__).parent


def _compat_embedding_parallel():
    """peft 0.19's resume path does `from transformers.integrations.tensor_parallel
    import EmbeddingParallel`, a symbol added only in transformers 4.58 (we pin
    4.57.6 via vllm 0.19). Inject a sentinel class so checkpoint resume works —
    the trainer never uses TP, so isinstance() against it is correctly False."""
    import transformers.integrations.tensor_parallel as tp

    if not hasattr(tp, "EmbeddingParallel"):
        class EmbeddingParallel:
            pass

        tp.EmbeddingParallel = EmbeddingParallel


def _force_nccl_all_reduce():
    """vLLM's custom all-reduce kernel crashes ('invalid argument' in
    custom_all_reduce.cuh) when the TP engine initializes inside trainer ranks
    that already hold CUDA contexts. TRL doesn't expose disable_custom_all_reduce,
    so inject it where TRL constructs the colocated engine."""
    import trl.generation.vllm_generation as _vg

    base = _vg.LLM

    class _PatchedLLM(base):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("disable_custom_all_reduce", True)
            super().__init__(*args, **kwargs)

    _vg.LLM = _PatchedLLM


MAX_EPISODE_TIMEOUT_S = 900  # tasks slower than this stall GRPO rollouts


def build_dataset(kernels_dir: str, repeats: int) -> Dataset:
    """One conversational prompt row per task, repeated (GRPO samples per row).

    Tasks without a baseline (invalid upstream reference, e.g. hotspot/srad) or
    with episode timeouts too long for rollouts are skipped with a warning.
    """
    rows, skipped = [], []
    for name, task in load_tasks(kernels_dir).items():
        if not task.get("baseline"):
            skipped.append((name, "no baseline"))
            continue
        if task.get("timeout_s", 600) > MAX_EPISODE_TIMEOUT_S:
            skipped.append((name, f"timeout_s {task['timeout_s']} > {MAX_EPISODE_TIMEOUT_S}"))
            continue
        prompt = [{"role": "user", "content": task["prompt"]}]
        rows.extend({"prompt": prompt, "task_id": name} for _ in range(repeats))
    for name, why in skipped:
        print(f"[build_dataset] skipping task '{name}': {why}")
    if not rows:
        raise RuntimeError(f"no usable tasks under {kernels_dir}")
    return Dataset.from_list(rows)


@record
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kernels_dir", default=str(PROJECT / "kernels"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--repeats", type=int, default=64, help="dataset rows per task")
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--vllm_gpu_mem", type=float, default=0.25)
    ap.add_argument("--vllm_tp", type=int, default=1,
                    help="TP for the colocated vLLM engine; use 4 for 7B so its "
                         "weights shard across GPUs instead of duplicating per rank")
    ap.add_argument("--vllm_mode", choices=["colocate", "server"], default="colocate",
                    help="server = dedicated trl vllm-serve process (1 GPU) + trainer "
                         "on the rest; the robust choice for 7B on 40GB cards")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tune (32B tier)")
    args = ap.parse_args()

    colocate_kwargs = {}
    if args.vllm_mode == "server":
        # Explicit loopback: TRL's default host is 0.0.0.0, which is NOT in
        # no_proxy on Polaris compute nodes — the ALCF proxy blackholes the
        # weight-sync handshake and rank 0 hangs forever.
        colocate_kwargs = dict(vllm_server_base_url="http://127.0.0.1:8000")
    if args.vllm_mode == "colocate":
        colocate_kwargs = dict(
            vllm_gpu_memory_utilization=args.vllm_gpu_mem,
            vllm_tensor_parallel_size=args.vllm_tp,
            vllm_enable_sleep_mode=True,
        )
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        **colocate_kwargs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        log_completions=True,
    )

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    _compat_embedding_parallel()
    if args.vllm_mode == "colocate" and args.vllm_tp > 1:
        _force_nccl_all_reduce()
    reward = HLSRewardFunc(args.kernels_dir)
    trainer = GRPOTrainer(
        model=args.model,
        args=cfg,
        reward_funcs=[reward],
        train_dataset=build_dataset(args.kernels_dir, args.repeats),
        peft_config=peft_config,
    )
    try:
        from transformers.trainer_utils import get_last_checkpoint

        last = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
        trainer.train(resume_from_checkpoint=last)
        trainer.save_model(args.output_dir + "/final")
    finally:
        reward.close()


if __name__ == "__main__":
    main()
