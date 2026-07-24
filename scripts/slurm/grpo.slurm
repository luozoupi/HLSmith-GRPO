#!/bin/bash
#SBATCH --job-name=grpo_hls
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=grpo_hls.%j.out
# Add your site's account/partition, e.g.:
##SBATCH --account=<project>
##SBATCH --partition=<gpu-partition>

# Slurm equivalent of scripts/pbs/grpo.pbs. See REPRODUCE.md.
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-Coder-7B-Instruct,\
# OUTDIR=$HLSMITH_ROOT/runs/grpo_7b_base_v1,GRPO_VLLM_MODE=server,\
# EXTRA="--max_steps 100 --save_steps 10 --lora --learning_rate 1e-5 \
#        --max_completion_length 1536 --kernels_dir $PWD/kernels_c2hls \
#        --per_device_train_batch_size 1 --gradient_accumulation_steps 16" \
#     scripts/slurm/grpo.slurm
#
# Budget walltime from HOURS, not steps: ~16-17h for 100 steps from a base model,
# ~6.5h from an SFT warmstart (base models emit more non-compiling code, so more
# rollouts burn the full synthesis timeout).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/scripts -> <repo>
REPO="$(dirname "$REPO")"
source "$REPO/scripts/env/env_llmhls.sh"
source "$REPO/scripts/env/env_vitis.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} WANDB_MODE=offline
# colocated vLLM + trainer fragment the allocator; expandable segments reclaims it
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
OUTDIR=${OUTDIR:-$HLSMITH_ROOT/runs/grpo_7b_base_v1}

# HLS reward workers. Scratch MUST be node-local (NVMe/tmpfs) — synthesis creates
# thousands of small files and will thrash a shared parallel filesystem.
SCRATCH="${HLSMITH_SCRATCH:-${SLURM_TMPDIR:-/tmp/$USER}/hlsenv.${SLURM_JOB_ID:-0}}"
mkdir -p "$SCRATCH"
export HLSENV_SCRATCH=$SCRATCH
export HLSENV_WORKERS=${HLSENV_WORKERS:-16}   # total per node, split across ranks
export HLSENV_CORES=${HLSENV_CORES:-12-31}    # cores 0-11 stay with training + vLLM
export HLSENV_ARCHIVE=${HLSENV_ARCHIVE:-$HLSMITH_ROOT/hls_archive/${SLURM_JOB_ID:-manual}}
mkdir -p "$HLSENV_ARCHIVE"

# Node-local scratch is wiped at job end — pull anything still there back.
trap 'mkdir -p "$HLSENV_ARCHIVE/scratch_leftover" && rsync -a "$SCRATCH/" "$HLSENV_ARCHIVE/scratch_leftover/" || echo "WARN: scratch rsync failed"' EXIT

# Optional: merge an SFT LoRA adapter into MERGE_BASE to produce MODEL. Do this on
# the compute node — a login-node cgroup will OOM-kill the ~30GB CPU merge.
if [ -n "${MERGE_ADAPTER:-}" ]; then
  python "$REPO/scripts/merge_adapter.py" \
    --base "${MERGE_BASE:-Qwen/Qwen2.5-Coder-7B-Instruct}" \
    --adapter "$MERGE_ADAPTER" --out "$MODEL"
fi

if [ "${GRPO_VLLM_MODE:-server}" = "server" ]; then
  # Server split: last GPU serves rollouts via `trl vllm-serve`; the rest train.
  # Colocate does NOT fit a 7B on 4x40GB (trainer + engine per GPU).
  NGPU=${SLURM_GPUS_ON_NODE:-4}
  SERVE_GPU=$((NGPU - 1))
  TRAIN_GPUS=$(seq -s, 0 $((NGPU - 2)))
  CUDA_VISIBLE_DEVICES=$SERVE_GPU trl vllm-serve --model "$MODEL" --port 8000 \
    --gpu-memory-utilization 0.90 > "trl_vllm_serve_${SLURM_JOB_ID:-0}.log" 2>&1 &
  SERVE_PID=$!
  trap 'kill $SERVE_PID 2>/dev/null || true' EXIT
  for i in $(seq 1 90); do
    curl -s --max-time 5 http://localhost:8000/health >/dev/null 2>&1 && { echo "vllm-serve ready (~$((i*10))s)"; break; }
    kill -0 $SERVE_PID 2>/dev/null || { echo "trl vllm-serve died — see trl_vllm_serve log"; exit 1; }
    sleep 10
  done
  # NOTE: vllm_server_base_url must be 127.0.0.1, not 0.0.0.0 — the latter gets
  # captured by some sites' egress proxy.
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch --num_processes $((NGPU - 1)) \
    "$REPO/train_grpo.py" --model "$MODEL" --output_dir "$OUTDIR" --vllm_mode server ${EXTRA:-}
else
  accelerate launch --num_processes "${SLURM_GPUS_ON_NODE:-4}" \
    "$REPO/train_grpo.py" --model "$MODEL" --output_dir "$OUTDIR" ${EXTRA:-}
fi
