# llm-hls — LLM + Vitis HLS RL pipeline on ALCF Polaris

A vLLM-served LLM writes HLS C/C++ kernels; **Vitis HLS 2023.2** (classic `vitis_hls` batch
flow) synthesizes them; QoR (latency / II / LUT / FF / DSP / BRAM) becomes the reward for
**TRL GRPO** training. SFT is supported via `train_sft.py`.

## Layout (everything under /eagle/argonne_tpc/lyb)

| Path | What |
|---|---|
| `tools/xilinx/` | Vitis/Vivado 2023.2 install root (Vivado ML Standard, free) |
| `envs/llmhls/` | venv: torch 2.10 (cu128) + vllm 0.19.0 + trl 1.8.0 (`requirements.lock`) |
| `caches/hf/` | HF_HOME — prefetched model weights (jobs run `HF_HUB_OFFLINE=1`) |
| `projects/llm-hls/` | this repo |
| `hls_archive/` | per-episode synth reports rsync'd back from node-local NVMe |
| `runs/` | checkpoints + tensorboard |

## Daily use

```bash
source scripts/env/env_llmhls.sh   # python env (kills ~/miniconda3 shadowing, sets caches/proxy)
source scripts/env/env_vitis.sh    # vitis_hls on PATH
```

- Unit tests (no Vitis needed): `python -m pytest tests/test_hlsenv.py -q`
- HLS integration tests: `python -m pytest tests/test_hls_integration.py -v`
- Baseline a task (required before GRPO): `python -m hlsenv baseline kernels/fir`
- Score any kernel: `python -m hlsenv synth kernels/fir --kernel my_kernel.cpp`

## Jobs (PBS, see scripts/pbs/)

| Script | Purpose |
|---|---|
| `compute_check.sh` | M4 validation inside an interactive debug job |
| `vllm_smoke.pbs` | serve 7B / 32B-AWQ / 70B-AWQ, record tok/s |
| `sft.pbs` | SFT (FSDP, 4 GPUs); gate with `--max_steps 100` on debug first |
| `grpo.pbs` | GRPO: colocated vLLM + 16 HLS workers on cores 12-31 |

Interactive node: `qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A argonne_tpc`

## Adding a task

Create `kernels/<name>/` with `kernel.cpp` (reference impl), `tb.cpp` (golden testbench,
exit 0 = pass), `spec.md` (the LLM prompt), `task.json` (top/part/clock/budgets). Then run
`python -m hlsenv baseline kernels/<name>`. The csim gate against `tb.cpp` is what prevents
reward hacking — keep testbenches strict.

## Reward ladder

NO_CODE / BANNED (`__SYNTHESIS__` escape hatch) −1.0 · COMPILE_FAIL/SYNTH_FAIL/TIMEOUT −0.8 ·
CSIM_FAIL −0.6 · OK: 0.2 + 0.6·min(2, baseline_lat/lat) − 0.5·(resource overage),
clamped to [−0.4, 2] — the floor keeps every correct design above every failure mode.
Positive reward REQUIRES passing csim against the golden testbench.
`HLSENV_WORKERS` is the total per node; hlsenv splits workers and `HLSENV_CORES` across ranks.
