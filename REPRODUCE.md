# GRPO-on-HLS: full reproduction recipe

Reproduces the `grpo_7b_base_v1` run (PBS job 7258561, exit 0, **16h44m**, 100 steps) —
GRPO'ing Qwen2.5-Coder-7B against **Vitis HLS QoR as the reward**.

Verified on ALCF Polaris (SLES 15 SP6, 4x A100-40GB/node, PBS Pro). Anything node-specific is
flagged **[SITE]**.

---

## 0. What the loop actually does

```
  TRL GRPOTrainer                          hlsenv (reward)
  ---------------                          ---------------
  prompt = spec.md  --> vLLM rollout  -->  extract ```cpp fence
  (translate plain C     (N=8 per          -> vitis_hls csim  (correctness gate)
   -> optimized HLS)      prompt)          -> vitis_hls csynth (QoR: latency/LUT/FF/DSP/BRAM)
                                           -> shaped reward vs per-task baseline
       ^                                             |
       +---------------------- advantage ------------+
```
One "task" = one benchmark directory holding the prompt, a golden testbench, and a
**pre-synthesized baseline QoR** that the reward measures speedup against.

---

## 1. Hardware / OS assumptions

| | |
|---|---|
| GPUs | 4x A100-40GB (1 node). Server mode: GPU3 serves rollouts, GPUs 0-2 train. |
| CPUs | 32+ cores. HLS synthesis is CPU-bound and runs *concurrently* with training. |
| Scratch | Node-local NVMe (`/local/scratch/$JOBID`). **Never** synthesize on Lustre/NFS — small-file thrash. |
| Disk | ~80GB Vitis install, ~60GB HF cache, ~30GB checkpoints. |

**[SITE]** Polaris specifics baked into the scripts: `/eagle/...` paths, PBS directives,
`proxy.alcf.anl.gov:3128` egress proxy on compute nodes, and `~/miniconda3` auto-activation that
must be scrubbed (see `env_llmhls.sh`).

---

## 2. Python environment (exact pins — these matter)

```bash
python -m venv $ROOT/envs/llmhls
source $ROOT/envs/llmhls/bin/activate
pip install "vllm==0.19.0"                 # pulls torch 2.10.0 + CUDA libs
pip install "trl[vllm]==1.8.0" "peft>=0.17" datasets accelerate tensorboard
pip freeze > requirements.lock
```

Confirmed working set:
```
vllm 0.19.0 · trl 1.8.0 · torch 2.10.0 · transformers 4.57.6
peft 0.19.1 · accelerate 1.14.0 · datasets 5.0.0 · tensorboard 2.21.0
```
Full lock: `envs/llmhls/requirements.lock`.

> **Version trap:** TRL 1.8 pins `vllm>=0.16,<=0.23`. Do not upgrade vLLM independently.
> No flash-attn build needed — vLLM ships kernels; HF side uses SDPA.

### Two required source patches (in `train_grpo.py`, applied automatically)
1. **`_compat_embedding_parallel()`** — peft 0.19's resume path imports `EmbeddingParallel`
   from `transformers.integrations.tensor_parallel`, a symbol that only exists in
   transformers >= 4.58 (we're pinned to 4.57.6 by vllm 0.19). Injects a sentinel class so
   **checkpoint resume doesn't crash**. Essential on a preemptible queue.
2. **`_force_nccl_all_reduce()`** — vLLM's custom all-reduce kernel crashes
   (`invalid argument` in `custom_all_reduce.cuh`) when a TP engine initializes inside trainer
   ranks that already hold CUDA contexts. Only needed for `colocate` + `vllm_tp>1`.

---

## 3. Vitis HLS 2023.2

Install Vivado ML Standard (free, no license) headless; `vitis_hls` classic flow is a bundled
component. Then `source scripts/env/env_vitis.sh`, which:
- sources **every** `settings64.sh` found (the 2023.2 unified installer puts classic Vitis HLS in
  a sibling tree `Vitis_HLS/2023.2` whose bin dir Vivado's own settings64.sh does *not* add),
- sets `XILINX_LOCAL_USER_DATA=no` so parallel workers don't race on `$HOME/.Xilinx`.

Target used throughout: **part `xcu280-fsvh2892-2L-e`, clock 3.33 ns**.

> **[SITE] 2023.2 officially supports SLES 15 SP3/SP4**; it runs fine on SP6 for the *batch*
> flow. If a node breaks: (1) missing `libtinfo.so.5` -> symlink ncurses6 into `compat_libs/`;
> (2) `GLIBCXX_*` errors -> rename the bundled `libstdc++.so.6*` under `Vivado/2023.2/lib/lnx64.o/`
> so the system one wins; (3) last resort -> Apptainer on an Ubuntu 22.04 base.
> Older repos may call `vitis-run --tcl` (2025.2 unified CLI); on 2023.2 use `vitis_hls -f`.

---

## 4. Build the task set

Each task dir needs: `spec.md` (the prompt), `kernel.cpp` (reference = baseline QoR source),
`tb.cpp` (golden testbench), header/aux files, and `task.json`.

```bash
python scripts/make_c2hls_tasks.py          # -> kernels_c2hls/<bench>/
# then baseline EVERY task (synthesizes kernel.cpp, writes QoR into task.json):
for d in kernels_c2hls/*/; do python -m hlsenv baseline "$d"; done
```
Baselining needs **both** env scripts sourced (`hlsenv/runner.py` checks `vitis_hls` is literally
on PATH). It's CPU-only — no GPU, no allocation. ~1-3 min/benchmark.

`task.json` fields: `top`, `part`, `clock_period_ns`, `timeout_s`, `csim`, `baseline` (filled by
the baseline step), `budgets` (50% of device: lut 651840, ff 1303680, dsp 4512, bram 2016),
`aux_files`.

> **Prompt/top-function trap.** `build_dataset` skips any task whose `baseline` is null. The
> prompt template defaults to a `workload()` top; if your benchmark's testbench calls something
> else (e.g. polybench `kernel_2mm`), the prompt **must** say so or generated kernels won't link
> against the testbench -> csim always fails -> zero reward -> nothing learns. Use the pipeline's
> `_build_benchmark_context()`, which emits ``Required HLS wrapper top function: `<top>` ``.
> Tasks exceeding `MAX_EPISODE_TIMEOUT_S=900` are also dropped (they stall rollouts).

---

## 5. The reward (`hlsenv/reward.py`)

Ladder — **the csim gate is mandatory for any positive reward** (anti reward-hacking):

| Outcome | Reward |
|---|---|
| NO_CODE / BANNED (no ```cpp fence, or uses `__SYNTHESIS__` escape hatch) | **-1.0** |
| SYNTH_FAIL (compiles+simulates, `csynth_design` fails) | **-0.8** |
| CSIM_FAIL (compiles, wrong vs golden testbench) | **-0.6** |
| DEGENERATE (csim-passes but empty/do-nothing circuit) | **-0.6** |
| OK | shaped in **[-0.4, 2.0]** |

```
reward = 0.2 (base) + 0.6 * min(2.0, baseline_latency / latency) - 0.5 * resource_overage
clamped to [-0.4, 2.0]
```

Three anti-hacking guards, all load-bearing:
- **`BANNED_TOKENS = ("__SYNTHESIS__",)`** — the canonical way to fool a csim-gated QoR reward is
  to `#ifdef` the kernel body away so csim runs the C path while HLS synthesizes nothing.
- **Degeneracy check** — a csim-passing *empty* circuit would otherwise collect a huge fake speedup.
- **`SPEEDUP_CAP = 2.0`** — bounds the incentive to game latency.

Synthesis runs with `set_param general.maxThreads 1`, in its own process group
(`start_new_session=True`) so a timeout kills the whole `vitis_hls` tree, not just the parent.

---

## 6. Launch (the exact proven configuration)

```bash
qsub -q preemptable -l walltime=24:00:00 -l select=1 -l filesystems=home:eagle -A <project> \
  -v MODEL=Qwen/Qwen2.5-Coder-7B-Instruct,\
OUTDIR=$ROOT/runs/grpo_7b_base_v1,\
GRPO_VLLM_MODE=server,\
EXTRA='--max_steps 100 --save_steps 10 --lora --learning_rate 1e-5 \
       --max_completion_length 1536 --kernels_dir $ROOT/projects/llm-hls/kernels_c2hls \
       --per_device_train_batch_size 1 --gradient_accumulation_steps 16' \
  scripts/pbs/grpo.pbs
```

Resolved hyperparameters:

| Knob | Value | Why |
|---|---|---|
| vLLM mode | **server** | Colocate does **not** fit 7B on 4x40GB (trainer + engine per GPU; sleep-mode cumem OOM). GPU3 serves, GPUs 0-2 train. |
| LoRA | **r=32, alpha=64, dropout=0.05** on all q/k/v/o/gate/up/down | Full-FT 7B OOMs on 4x40GB. |
| learning_rate | **1e-5** | Tuned for LoRA. The script *default is 1e-6* — 10x too low here; pass it explicitly. |
| max_completion_length | **1536** | 2048 caused late-run OOM; median completion is ~509 tok so it rarely binds. |
| per_device_batch / accum | **1 / 16** | Global batch 3x1x16 = 48, divisible by `num_generations=8`. |
| num_generations | 8 (default) | GRPO group size. |
| repeats | 64 (default) | Dataset rows per task. |
| max_steps / save_steps | **100 / 10** | save_steps=10 keeps preemption loss small. `save_total_limit=3`. |
| bf16, gradient_checkpointing | True | |

`vllm_server_base_url` must be **`http://127.0.0.1:8000`**, not `0.0.0.0` — the latter gets
captured by the site proxy.

### HLS worker knobs (set by `grpo.pbs`)
```
HLSENV_SCRATCH=/local/scratch/$JOBID/hlsenv   # node-local NVMe
HLSENV_WORKERS=16                             # total per node, split across ranks
HLSENV_CORES=12-31                            # cores 0-11 reserved for training + vLLM
HLSENV_ARCHIVE=$ROOT/hls_archive/$JOBID       # reports rsync'd out (NVMe is wiped at job end)
```
Calibrate `HLSENV_WORKERS` so reward throughput >= rollout demand; otherwise GPUs idle waiting
on synthesis. ~240-900 syntheses/h on 20 cores.

---

## 7. Expected behaviour

- **Runtime ~16-17h for 100 steps from base.** From an SFT warmstart it's **~6.5h** — the base
  model emits more non-compiling junk, so more rollouts burn the full synthesis timeout.
  Budget walltime from *this* number, not from step count.
- Reward climbs roughly **-0.45 -> -0.18** over 100 steps; `frac_reward_zero_std` should be ~0
  (if it pins at 1.0, every generation in a group scores identically = no learning signal).
- Checkpoints every 10 steps; **resume is automatic** via `get_last_checkpoint(output_dir)`.
- Merge the adapter for eval: `scripts/merge_adapter.py --base <base> --adapter <ckpt> --out <dir>`.
  Run it **on a compute node** — a login-node cgroup will OOM-kill the ~30GB CPU merge.

## 8. Gate before committing a long run
Do a 2-step run on a debug queue with a 1.5B model first
(`-v MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct,EXTRA="--max_steps 2"`). Check: non-degenerate reward
spread, GPUs not idling on rewards, checkpoint save/resume round-trips. That gate caught the
colocate-OOM and the custom-all-reduce crash before they cost a 24h slot.

## 9. Evaluate
Serve the merged model with vLLM and drive the agentic pipeline against it
(`scripts/pbs/c2hls_vllm.pbs`). **Report cosim, not csim** — csim systematically overstates
correctness (one arm went 23/26 csim -> 5/26 cosim). See `RESULTS.md`.
