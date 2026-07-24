# CUDA→CSL offline SFT on Polaris (no Cerebras, no commercial API)

Distill the historical `xkernel` CUDA→Cerebras-CSL translation corpus (produced by
commercial/hosted models on a real WSE simulator) into a **local open-weights model**
via offline SFT, serve it with **vLLM**, and proxy-evaluate it against the commercial
artifacts — entirely on Polaris A100s. **No Cerebras SDK exists here**, so real
`cslc`/`cs_python` QoR is out of scope; evaluation is a pure-Python proxy (with a
staged export for later real-QoR scoring on a Cerebras machine).

Base model: **Qwen/Qwen3-32B** (cached; dense; native 40960 context — 32K fits with no
RoPE scaling). Alternative: `google/gemma-4-31B-it` (serve with `USE_G4_ENV=1`).

## Layout
- Corpus (extracted): `/eagle/argonne_tpc/lyb/data/csl_sft/raw/corpus/`
- Processed data:     `/eagle/argonne_tpc/lyb/data/csl_sft/{train,val}.jsonl (+ .meta.jsonl), test_prompts.jsonl, prep_summary.json`
- Harness worktree:   `/eagle/argonne_tpc/lyb/xkernel-v1_0/` (branch origin/v1_0; harness + kernel bundles)
- Scripts:            `/eagle/argonne_tpc/lyb/projects/llm-hls/scripts/`
- Runs/outputs:       `/eagle/argonne_tpc/lyb/runs/`

## Data format & the completion-only-loss trick
Records are emitted as **prompt-completion**:
`{"prompt":[system, "Translate ..." user], "completion":[assistant CSL]}`.
TRL 1.8 `SFTTrainer` auto-detects this and sets `completion_only_loss=True`, so the loss
is computed only on the ~2.5K-token CSL, not the ~20K-token reference-bundle prompt.
`train_sft.py` needs **no change**. (Prep verified: prompt=20K tok masked, completion=2.4K tok in loss.)

Prep results (32K filter, Qwen3 tokenizer):
6066 → 2440 pass → 2162 reconstructable → 163 held-out (6 kernels) → 1999 pool → −186 over-32K →
**1632 train / 181 val**. Held-out kernels (eval-only, never in train):
GEMM, Cholesky, Jacobi-2D-5pt, SpMV-CSR, Tensor-Transpose-021, Residual.

## Step 0 — (re)build the processed data  [login node, no GPU]
```bash
module use /soft/modulefiles && module load conda
source /eagle/argonne_tpc/lyb/projects/llm-hls/scripts/env/env_llmhls.sh
cd /eagle/argonne_tpc/lyb/data/csl_sft/raw && tar xzf /home/lyb/corpus_sft_dpo.tar.gz   # if not extracted
cd /eagle/argonne_tpc/lyb/projects/llm-hls
python scripts/prep_csl_corpus.py \
  --sft-jsonl /eagle/argonne_tpc/lyb/data/csl_sft/raw/corpus/sft_corpus.jsonl \
  --out-dir   /eagle/argonne_tpc/lyb/data/csl_sft \
  --model     Qwen/Qwen3-32B --max-seq-len 32768
```

## Step 1 — SFT  (reuses train_sft.py + configs/fsdp.yaml + scripts/pbs/sft.pbs)
```bash
cd /eagle/argonne_tpc/lyb/projects/llm-hls
# 1a. pipeline smoke (cheap model, 20 steps) — validates the whole training path.
# Use the full 1h debug walltime: TRL tokenizes the ENTIRE dataset up front (~1632 x 27K tok)
# before training, which is slow on the FIRST run; it is cached afterwards (HF datasets cache),
# and each new tokenizer (e.g. Qwen3) pays it once. VALIDATED 2026-07-21: loss 1.015->1.009,
# completion-only loss active, adapter saved at csl_sft_smoke/checkpoint-20/.
qsub -q debug -l walltime=01:00:00 -v \
 MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct,\
DATASET=/eagle/argonne_tpc/lyb/data/csl_sft/train.jsonl,\
OUTDIR=/eagle/argonne_tpc/lyb/runs/csl_sft_smoke,\
EXTRA="--lora --max_seq_len 32768 --max_steps 20" scripts/pbs/sft.pbs

# 1b. gate (VALIDATED 2026-07-21): Qwen3-8B @ 32K, HF grad-ckpt + fsdp_gc.yaml.
# Fits 4x A100-40GB, ~132 s/step, loss 0.94. (Qwen3-32B does NOT fit at 32K on 40GB.)
qsub -q debug -l walltime=01:00:00 -v \
 MODEL=Qwen/Qwen3-8B,DATASET=/eagle/argonne_tpc/lyb/data/csl_sft/train.jsonl,\
OUTDIR=/eagle/argonne_tpc/lyb/runs/csl_sft_qwen3_8b_gate,\
ACCEL_CONFIG=/eagle/argonne_tpc/lyb/projects/llm-hls/configs/fsdp_gc.yaml,\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,\
EXTRA="--lora --grad-ckpt --max_seq_len 32768 --max_steps 10" scripts/pbs/sft.pbs

# 1c. full run (preemptable; ~153 steps for 3 epochs @ ~132 s/step ~= 5.6 h)
qsub -q preemptable -l walltime=12:00:00 -v \
 MODEL=Qwen/Qwen3-8B,DATASET=/eagle/argonne_tpc/lyb/data/csl_sft/train.jsonl,\
OUTDIR=/eagle/argonne_tpc/lyb/runs/csl_sft_qwen3_8b_v1,\
ACCEL_CONFIG=/eagle/argonne_tpc/lyb/projects/llm-hls/configs/fsdp_gc.yaml,\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,\
EXTRA="--lora --grad-ckpt --num_train_epochs 3 --learning_rate 1e-5 --max_seq_len 32768 --save_steps 25" \
 scripts/pbs/sft.pbs
```
Adapter → `<OUTDIR>/final` (directly usable; FULL_STATE_DICT). `ACCEL_CONFIG` selects the accelerate
config; `--grad-ckpt` turns on HF gradient checkpointing (required for long context — FSDP2's own
`fsdp_activation_checkpointing` did NOT reduce the activation spike, so `fsdp_gc.yaml` disables it).

**Why 8B, not 32B:** the OOM at 32K was the forward *activation* spike, not params (memory probe:
sharded baseline only 4.18 GB/GPU). Qwen3-32B/gemma-4-31B still exceed 40GB even with HF grad-ckpt;
Qwen3-8B (same Qwen3 tokenizer, so no data rework) fits 32K with headroom. Fallbacks if needed:
`--max_seq_len 24576`, QLoRA (`pip install bitsandbytes`), or `Qwen/Qwen2.5-Coder-14B-Instruct`.

## Step 2 — Serve + proxy-evaluate  (scripts/pbs/csl_eval.pbs)
Serves base + LoRA on one node, generates CSL for the 6 held-out kernels (no SDK), and
scores fine-tuned vs base vs commercial artifact.
```bash
# smoke (1.5B: MAXLEN must be <=32768; use the checkpoint-N dir if /final wasn't reached)
qsub -q debug -l walltime=00:30:00 -v \
 BASE_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct,TP=1,MAXLEN=32768,MAX_TOKENS=4096,\
ADAPTER=/eagle/argonne_tpc/lyb/runs/csl_sft_smoke/checkpoint-20,\
OUTDIR=/eagle/argonne_tpc/lyb/runs/eval_smoke,MAX_PER_KERNEL=2 scripts/pbs/csl_eval.pbs

# real (Qwen3-8B: native 40960 ctx; TP=2 is enough for an 8B, MAXLEN=40960)
qsub -q preemptable -l walltime=02:00:00 -v \
 BASE_MODEL=Qwen/Qwen3-8B,TP=2,MAXLEN=40960,MAX_TOKENS=6000,\
ADAPTER=/eagle/argonne_tpc/lyb/runs/csl_sft_qwen3_8b_v1/final,\
OUTDIR=/eagle/argonne_tpc/lyb/runs/eval_qwen3_8b,MAX_PER_KERNEL=8 scripts/pbs/csl_eval.pbs
```
Outputs in `<OUTDIR>`: `gen_finetuned.jsonl`, `gen_base.jsonl`, `scored_*.jsonl`,
`scoreboard.json` (per-kernel + overall: gen_rate, syntax_valid, contract_ok, code_sim,
exact_match), and `csl_generations_for_real_qor.tar.gz` (staged for a Cerebras-SDK machine).
Success signal: **fine-tuned ≥ base** on the proxy metrics, approaching the commercial ceiling.

## New scripts (this work)
- `scripts/prep_csl_corpus.py`  — corpus → prompt-completion SFT data + held-out test set
- `scripts/gen_csl_heldout.py`  — generate CSL vs local vLLM (replays commercial prompts; no SDK)
- `scripts/score_csl_proxy.py`  — pure-python proxy scoreboard + real-QoR export bundle
- `scripts/pbs/csl_eval.pbs`     — serve + generate + score in one PBS job
Reused unchanged: `train_sft.py`, `configs/fsdp.yaml`, `scripts/pbs/sft.pbs`,
`scripts/env/env_llmhls.sh`, `scripts/merge_adapter.py`.

## Scope boundary
Proxy metrics are **not** real correctness/`cycles_send` (no Cerebras SDK on Polaris).
The export tarball lets you run real `benchmark_csl.py` (cslc + cs_python) later on a
Cerebras-SDK host. Optional next step: DPO on the 2249 pass-vs-fail pairs in
`dpo_corpus.jsonl` (reconstruct the same way; add a `train_dpo.py` with TRL DPOTrainer).
```
