# HLSmith-GRPO — RL/SFT for LLM-generated HLS, rewarded by Vitis HLS QoR

A vLLM-served LLM writes HLS C/C++ kernels; **Vitis HLS** synthesizes them; QoR
(latency / II / LUT / FF / DSP / BRAM) becomes the reward for **TRL GRPO** training.
SFT is supported via `train_sft.py`.

```
  TRL GRPOTrainer                          hlsenv (reward)
  ---------------                          ---------------
  prompt = spec.md  --> vLLM rollout  -->  extract ```cpp fence
  (plain C -> optimized HLS)  (N per        -> vitis_hls csim   (correctness gate)
                               prompt)      -> vitis_hls csynth (QoR)
       ^                                    -> shaped reward vs per-task baseline
       +---------------------- advantage ---------------+
```

Runs on any HPC node with GPUs + a Vitis HLS install. **[REPRODUCE.md](REPRODUCE.md) is the
full recipe** — pinned versions, exact hyperparameters of a proven run, and the traps.

## Quick start

```bash
git clone <this repo> && cd HLSmith-GRPO
cp scripts/env/site.sh.example scripts/env/site.sh    # edit: paths, Vitis, scratch
source scripts/env/env_llmhls.sh                      # python env
source scripts/env/env_vitis.sh                       # vitis_hls on PATH

python -m pytest tests/test_hlsenv.py -q               # unit tests (no Vitis needed)
python -m hlsenv baseline kernels_c2hls/aes            # baseline one task
python -m hlsenv synth  kernels_c2hls/aes --kernel my_kernel.cpp   # score a kernel
```

Everything site-specific lives in `scripts/env/site.sh` (untracked; see
`site.sh.example`). Without one, paths are inferred from the repo location.

## Layout

| Path | What |
|---|---|
| `train_grpo.py` / `train_sft.py` | training entrypoints |
| `hlsenv/` | reward package: sandboxed `vitis_hls` runner, QoR parser, shaped reward, task loader |
| `kernels_c2hls/` | 17 rodinia/ML4Accel GRPO tasks |
| `kernels_polybench/` | 22 polybench GRPO tasks |
| `kernels_c2hls_steps/` | per-transformation step-level task variants |
| `scripts/` | task builders, env setup, job scripts, analysis/comparison tooling |
| `scripts/pbs/`, `scripts/slurm/` | batch job scripts (PBS and Slurm) |

Outputs (`runs/`, `caches/`, `models/`, `hls_archive/`) live under `$HLSMITH_ROOT` and are
gitignored.

## Training

See REPRODUCE.md for the full command. In short, from a base model on 4 GPUs:

```bash
# Slurm
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-Coder-7B-Instruct,GRPO_VLLM_MODE=server,\
OUTDIR=$HLSMITH_ROOT/runs/grpo_v1,\
EXTRA="--max_steps 100 --save_steps 10 --lora --learning_rate 1e-5 \
       --max_completion_length 1536 --kernels_dir $PWD/kernels_c2hls \
       --per_device_train_batch_size 1 --gradient_accumulation_steps 16" \
  scripts/slurm/grpo.slurm

# PBS: same variables via qsub -v ... scripts/pbs/grpo.pbs
```

Budget walltime from **hours, not steps**: ~16-17h for 100 steps from a base model,
~6.5h from an SFT warmstart. Checkpoints every 10 steps; resume is automatic.

## Adding a task

Create `kernels_<suite>/<name>/` with `kernel.cpp` (reference impl, also the baseline QoR
source), `tb.cpp` (golden testbench, exit 0 = pass), `spec.md` (the LLM prompt), and
`task.json` (top / part / clock / budgets). Then:

```bash
python -m hlsenv baseline kernels_<suite>/<name>
```

Tasks without a baseline are silently skipped by `build_dataset`. The csim gate against
`tb.cpp` is what prevents reward hacking — keep testbenches strict.

**Make sure `spec.md` names the same top function your testbench calls.** If the prompt asks
for `workload()` but the testbench calls `kernel_2mm`, nothing links, csim always fails,
every reward is negative, and training silently learns nothing.

## Reward ladder

| Outcome | Reward |
|---|---|
| NO_CODE / BANNED (`__SYNTHESIS__` escape hatch) | −1.0 |
| COMPILE_FAIL / SYNTH_FAIL / TIMEOUT | −0.8 |
| CSIM_FAIL, or DEGENERATE (csim-passing but empty circuit) | −0.6 |
| OK | `0.2 + 0.6·log2(min(8, baseline_lat/lat)) − 0.5·overage`, clamped to [−0.4, 2] |

**Log-scale speedup** (`+0.6` per doubling vs baseline): match=+0.2, 2×=+0.8, 4×=+1.4, 8×=+2.0.
This is deliberate — a linear term capped at 2× paid ~0.8 for merely *translating* the kernel and
only +0.6 more for a full 2×, so a small model learned to reproduce the clean reference and stop
(measured: ~1.0× gold everywhere, while frontier models hit 2–93× faster). log2 keeps a strong
gradient across the whole achievable range, so **deep optimization is what pays**.

The floor keeps every correct design above every failure mode. **Positive reward requires
passing csim against the golden testbench** — that gate, plus the banned-token and
degenerate-circuit checks, is what stops the model from optimizing the metric instead of
the kernel.

## Evaluating

Serve a merged model with vLLM and drive your agentic pipeline against it. **Report cosim,
not csim** — csim systematically overstates correctness (in our runs one arm went 23/26 csim
→ 5/26 cosim; the fast kernels were the broken ones).

`HLSENV_WORKERS` is the total per node; hlsenv splits workers and `HLSENV_CORES` across ranks.
