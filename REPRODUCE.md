# GRPO-on-HLS: full reproduction recipe

Reproduces a proven run: GRPO'ing Qwen2.5-Coder-7B against **Vitis HLS QoR as the reward**
(100 steps, 16h44m wall, 4x A100-40GB).

Written for a **generic HPC node**. Site-specific details are confined to
`scripts/env/site.sh` and Appendix A. Originally developed on ALCF Polaris.

---

## 0. What the loop does

One "task" = a directory holding the prompt, a golden testbench, and a **pre-synthesized
baseline QoR** the reward measures speedup against. Each rollout is extracted, csim'd for
correctness, csynth'd for QoR, and scored.

---

## 1. Requirements

| | Minimum | Notes |
|---|---|---|
| GPUs | 4x 40GB (A100 or better) | Server mode: 1 GPU serves rollouts, the rest train. 2 GPUs works with reduced batch. |
| CPU cores | 32+ | HLS synthesis is CPU-bound and runs *concurrently* with training. This is usually the throughput bottleneck, not the GPU. |
| Node-local scratch | ~50GB NVMe/tmpfs | **Must not** be a shared parallel FS (Lustre/GPFS/NFS) — synthesis creates thousands of small files. |
| Shared storage | ~150GB | Vitis (~80GB), HF cache (~60GB), checkpoints (~30GB). |
| Scheduler | Slurm or PBS | Scripts provided for both. |
| Vitis HLS | 2023.2 recommended | Any version works if `HLS_TCL_RUNNER` matches (see §3). |

No internet needed at runtime (jobs run `HF_HUB_OFFLINE=1`) — but you must prefetch model
weights first, from a node that does have egress.

---

## 2. Python environment (exact pins — these matter)

```bash
python -m venv $HLSMITH_ROOT/envs/llmhls
source $HLSMITH_ROOT/envs/llmhls/bin/activate
pip install "vllm==0.19.0"                 # pulls torch 2.10.0 + CUDA libs
pip install "trl[vllm]==1.8.0" "peft>=0.17" datasets accelerate tensorboard
```

Confirmed working set — `requirements.lock` has the full freeze:
```
vllm 0.19.0 · trl 1.8.0 · torch 2.10.0 · transformers 4.57.6
peft 0.19.1 · accelerate 1.14.0 · datasets 5.0.0
```

> **Version trap:** TRL 1.8 pins `vllm>=0.16,<=0.23`. Don't upgrade vLLM independently.
> Match the torch CUDA build to your driver. No flash-attn build needed.

### Two required source patches (already in `train_grpo.py`)
1. **`_compat_embedding_parallel()`** — peft 0.19's resume path imports `EmbeddingParallel`
   from `transformers.integrations.tensor_parallel`, which only exists in transformers >= 4.58
   (vllm 0.19 pins 4.57.6). Injects a sentinel so **checkpoint resume doesn't crash** —
   essential on a preemptible queue.
2. **`_force_nccl_all_reduce()`** — vLLM's custom all-reduce crashes when a TP engine
   initializes inside trainer ranks that already hold CUDA contexts. Only for `colocate` + `vllm_tp>1`.

---

## 3. Vitis HLS

Install Vivado ML Standard (free, no license); `vitis_hls` is a bundled component. Point
`XILINX_ROOT`/`XILINX_VERSION` at it in `site.sh`, or `module load` your site's Vitis before
sourcing `env_vitis.sh` — it detects either.

`env_vitis.sh` sources **every** `settings64.sh` it finds: the unified installer puts classic
Vitis HLS in a sibling tree (`Vitis_HLS/<ver>`) whose bin dir Vivado's own settings64.sh does
*not* add. It also sets `XILINX_LOCAL_USER_DATA=no` so parallel workers don't race on
`$HOME/.Xilinx`.

**Pick the right tcl runner** — this is the single most common portability break:
```bash
export HLS_TCL_RUNNER="vitis_hls -f"                    # classic, 2023.2 (default)
export HLS_TCL_RUNNER="vitis-run --tcl --input_file"    # unified CLI, 2024.1+
```

Set your target device in `site.sh` (`HLS_PART`, `HLS_CLOCK_NS`). The reference runs used
`xcu280-fsvh2892-2L-e @ 3.33ns`. **Whatever you choose must match what your baselines were
generated against** — otherwise every reward is silently skewed.

### Troubleshooting an unsupported OS
Vitis supports a narrow set of distros; the **batch** flow usually works anyway (GUI failure
modes don't apply). In order: (1) missing `libtinfo.so.5` -> symlink ncurses6 into
`$XILINX_ROOT/compat_libs/` (auto-added to `LD_LIBRARY_PATH`); (2) `GLIBCXX_*` errors ->
rename the bundled `libstdc++.so.6*` under `Vivado/<ver>/lib/lnx64.o/` so the system one wins;
(3) last resort -> Apptainer/Singularity on a supported base image, bind-mounting the install.

---

## 4. Build the task set

Task dirs are committed, but **baselines are Vitis-version and part specific — regenerate them**:

```bash
for d in kernels_c2hls/*/; do python -m hlsenv baseline "$d"; done
```
CPU-only, no GPU, ~1-3 min/benchmark. Needs **both** env scripts sourced (`hlsenv/runner.py`
checks `vitis_hls` is literally on PATH).

To build tasks from your own benchmarks, see `scripts/make_c2hls_tasks.py` /
`make_polybench_tasks.py`. Each task needs `spec.md` (prompt), `kernel.cpp` (reference =
baseline QoR source), `tb.cpp` (golden testbench), header/aux, and `task.json`.

> **Two silent-failure traps.**
> - `build_dataset` **skips any task whose `baseline` is null** — check the count it prints.
>   Tasks slower than `MAX_EPISODE_TIMEOUT_S` (900s) are dropped too, since they stall rollouts.
> - The prompt must name the **same top function the testbench calls**. If `spec.md` asks for
>   `workload()` but `tb.cpp` calls `kernel_2mm`, nothing links -> csim always fails -> every
>   reward is negative -> training burns a full allocation learning nothing.

---

## 5. The reward (`hlsenv/reward.py`)

**The csim gate is mandatory for any positive reward.**

| Outcome | Reward |
|---|---|
| NO_CODE / BANNED (`__SYNTHESIS__`) | -1.0 |
| SYNTH_FAIL | -0.8 |
| CSIM_FAIL / DEGENERATE | -0.6 |
| OK | `0.2 + 0.6*log2(min(8, baseline_lat/lat)) - 0.5*overage`, clamped to [-0.4, 2.0] |

Log-scale speedup (+0.6 per doubling: match=+0.2, 2x=+0.8, 4x=+1.4, 8x=+2.0). A linear term
capped at 2x under-rewarded optimization — a 7B learned to reproduce the reference (~1x gold) and
stop; log2 keeps a strong gradient across the ~8-90x achievable headroom so deep optimization pays.

Three load-bearing anti-reward-hacking guards:
- **`BANNED_TOKENS = ("__SYNTHESIS__",)`** — the canonical exploit is `#ifdef`-ing the kernel
  body away so csim runs the C path while HLS synthesizes nothing.
- **Degeneracy check** — a csim-passing *empty* circuit would otherwise collect a huge fake speedup.
- **`SPEEDUP_CAP = 2.0`** — bounds the incentive to game latency.

Synthesis runs with `set_param general.maxThreads 1` in its own process group
(`start_new_session=True`), so a timeout kills the whole `vitis_hls` tree.

---

## 6. Launch (the exact proven configuration)

```bash
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-Coder-7B-Instruct,\
OUTDIR=$HLSMITH_ROOT/runs/grpo_7b_base_v1,GRPO_VLLM_MODE=server,\
EXTRA="--max_steps 100 --save_steps 10 --lora --learning_rate 1e-5 \
       --max_completion_length 1536 --kernels_dir $PWD/kernels_c2hls \
       --per_device_train_batch_size 1 --gradient_accumulation_steps 16" \
  scripts/slurm/grpo.slurm
```
(PBS: same variables via `qsub -v ... scripts/pbs/grpo.pbs`.)

| Knob | Value | Why |
|---|---|---|
| vLLM mode | **server** | Colocate does **not** fit 7B on 4x40GB. Last GPU serves, rest train. |
| LoRA | **r=32, alpha=64, dropout=0.05**, all q/k/v/o/gate/up/down | Full-FT 7B OOMs on 4x40GB. |
| learning_rate | **1e-5** | Tuned for LoRA. **The script default is 1e-6 — 10x too low here. Pass it explicitly.** |
| max_completion_length | **1536** | 2048 caused late-run OOM; median completion ~509 tok so it rarely binds. |
| batch / accum | **1 / 16** | Global batch = ranks x 1 x 16 must be divisible by `num_generations` (8). |
| max_steps / save_steps | **100 / 10** | Frequent saves keep preemption loss small. `save_total_limit=3`. |

### HLS worker knobs
```
HLSENV_SCRATCH   node-local NVMe/tmpfs (job-scoped)
HLSENV_WORKERS   16     total per node, split across ranks
HLSENV_CORES     12-31  cores 0-11 reserved for training + vLLM
HLSENV_ARCHIVE   where synth reports are rsync'd (node-local scratch is wiped at job end)
```
Tune `HLSENV_WORKERS` so reward throughput >= rollout demand, else GPUs idle waiting on
synthesis. ~240-900 syntheses/h on 20 cores.

`vllm_server_base_url` must be **`127.0.0.1`**, not `0.0.0.0` — the latter gets captured by
some sites' egress proxy.

---

## 7. Smoke-test checklist (do this before a long run)

A 2-step run with a 1.5B model on a short/debug queue caught two failures that would each
have cost a full 24h slot:

```bash
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct,EXTRA="--max_steps 2 --lora" \
  scripts/slurm/grpo.slurm
```

- [ ] `vitis_hls -version` runs; a trivial tcl synthesizes
- [ ] `python -m pytest tests/test_hlsenv.py -q` passes
- [ ] Every task baselined — `build_dataset` reports the count you expect, none skipped
- [ ] One task scores end-to-end: `python -m hlsenv synth <task> --kernel <good>.cpp` -> positive;
      a deliberately broken kernel -> negative
- [ ] Rollouts produce the **correct top function name** (grep a completion)
- [ ] **`frac_reward_zero_std` ~= 0.** If it pins at 1.0, every generation in a group scores
      identically -> no gradient signal -> the run is worthless however long it trains
- [ ] GPUs not idling on rewards (raise `HLSENV_WORKERS` if they are)
- [ ] Checkpoint save **and resume** round-trip
- [ ] Node-local scratch is genuinely node-local

---

## 8. Expected behaviour

- **~16-17h per 100 steps from a base model; ~6.5h from an SFT warmstart.** Base models emit
  more non-compiling code, so more rollouts burn the full synthesis timeout. Budget walltime
  from this, not from step count.
- Reward climbs roughly **-0.45 -> -0.18** over 100 steps.
- Checkpoints every 10 steps; resume is automatic via `get_last_checkpoint(output_dir)`.
- Merge the adapter for eval: `scripts/merge_adapter.py --base <base> --adapter <ckpt> --out <dir>`.
  Run it **on a compute node** — a login-node cgroup will OOM-kill the ~30GB CPU merge.

## 9. Evaluate
Serve the merged model with vLLM and drive your agentic pipeline against it.
**Report cosim, not csim** — csim systematically overstates correctness (one arm went 23/26
csim -> 5/26 cosim, and the *fast* kernels were the broken ones).

---

## Appendix A — ALCF Polaris notes

The original site. Useful as a worked example of what `site.sh` must capture.

- **Filesystem:** `$HOME` is quota-limited; everything lives on `/eagle/<project>/<user>`
  (`HLSMITH_ROOT`). Synthesis on `/local/scratch/$PBS_JOBID` (node-local NVMe, wiped at job end).
- **Scheduler:** PBS Pro. `qsub -q preemptable -l select=1 -l walltime=24:00:00
  -l filesystems=home:eagle -A <project>`. The `preemptable` queue can kill jobs at any time
  (exit 143) — resume-safe checkpointing is not optional there.
- **Egress:** compute nodes have no direct internet; `env_llmhls.sh` auto-detects Polaris and
  sets the ALCF proxy. Login nodes have direct egress.
- **Conda shadowing:** a `~/miniconda3` base env auto-activates in `.bashrc` and shadows the
  venv's python. `env_llmhls.sh` deactivates and scrubs it (`HLSMITH_KEEP_CONDA=1` opts out).
- **Login-node thread cgroup:** importing torch/numpy on a login node spawns one OpenBLAS
  thread per core (64+), exceeds the cgroup, and **segfaults with a multi-GB core dump**.
  `env_llmhls.sh` caps threads to 4 outside batch jobs. If you find stray `core.*` files,
  this is where they came from.
- **OS:** SLES 15 SP6 — outside Vitis 2023.2's supported list, but the batch flow works.
  See §3 troubleshooting.

## Appendix B — porting checklist

1. `cp scripts/env/site.sh.example scripts/env/site.sh`; set `HLSMITH_ROOT`, `HLSMITH_VENV`,
   `XILINX_ROOT`/`XILINX_VERSION`, `HLSMITH_SCRATCH`, `HLS_PART`/`HLS_CLOCK_NS`, and
   `HLSMITH_PROXY` (set to `""` if your compute nodes have direct egress).
2. Build the venv from `requirements.lock`; match the torch CUDA build to your driver.
3. Set `HLS_TCL_RUNNER` for your Vitis version.
4. Prefetch model weights while you still have egress.
5. **Re-baseline every task** — committed baselines are from Vitis 2023.2 @ xcu280/3.33ns.
6. Adapt `scripts/slurm/grpo.slurm` (account, partition, GPU flags) or the PBS equivalent.
7. Run the §7 smoke test.

`task.json`'s `source_benchmark` field is inert provenance pointing at the original machine —
nothing reads it; ignore stale paths there.
