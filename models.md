# Model checkpoints

Weights are large and live under `$HLSMITH_ROOT/models/` (gitignored), not in the repo.
This file is the index. Regenerate any of these with the recipe in REPRODUCE.md.

## The SFT checkpoint the harness consumes

**`models/qwen7b_sft_v1/`** — offline SFT of Qwen2.5-Coder-7B-Instruct on the polybench
teacher corpus (`data/sft_qwen_v3`, **train split only** — see `splits.json`).

| Item | Path | Use |
|---|---|---|
| LoRA adapter | `models/qwen7b_sft_v1/adapter/` | r=32, alpha=64, base `Qwen/Qwen2.5-Coder-7B-Instruct` |
| **Merged model** | `models/qwen7b_sft_v1/merged/` | **This is what the harness uses** — 4 safetensor shards, servable by vLLM directly |

- **As a GRPO warmstart** (SFT->GRPO arm): pass the merged dir as `MODEL=...` (job 7258562
  warmstarted from exactly `models/qwen7b_sft_v1/merged`). Or merge on-node from the adapter
  via `MERGE_ADAPTER=.../adapter MERGE_BASE=Qwen/Qwen2.5-Coder-7B-Instruct`.
- **For evaluation**: serve the merged dir with `scripts/pbs/c2hls_vllm.pbs` (`MODEL=...`).

> The merged model is the harness-facing artifact. The raw adapter is kept only for
> re-merging against a different base or inspecting the LoRA deltas.

## Other trained checkpoints (context)

| Dir | What | From |
|---|---|---|
| `models/qwen7b_grpo_base_v1` | online GRPO (v1) from base, 100 steps | Qwen2.5-Coder-7B base, `kernels_c2hls` (rodinia) |
| `models/qwen7b_sft_grpo_v1` | SFT -> GRPO (v1), 100 steps | `qwen7b_sft_v1/merged` warmstart |
| `runs/grpo_polybench_v2/` | online GRPO-v2 (in progress) | 7B base, `kernels_polybench` (job 7276048) |

## Provenance note

The SFT corpus (`data/sft_qwen_v3`) mixes single-shot translation and multi-turn agentic
trajectories (465 / 3301 train rows respectively), all distilled from Sonnet/Haiku teachers
running the c2hls pipeline. Both are held out on the **same** benchmark split (`splits.json`),
so the SFT and GRPO-v2 val/test numbers are directly comparable.
