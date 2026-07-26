# Building GRPO properly for an HLS benchmark claim

How to train a GRPO model that is *benchmarked on polybench* and make a defensible claim
about it. Written from what went wrong in the first iteration.

## The two failure modes we actually hit

1. **Reward paid for translation, not optimization.** The old reward was
   `0.2 + 0.6*min(2, baseline/lat)`. A model that merely reproduced the clean reference kernel
   scored **0.8** (of a max 1.4), and the 2× cap meant real optimization added little. Result: the
   7B learned to translate to ~1.0× the reference and stop — while frontier models optimized the
   same kernels **2–93× faster**, all cosim-verified. The gap was an *optimization* gap the reward
   never asked to close.
   **Fix (shipped):** log-scale speedup, `0.2 + 0.6*log2(min(8, baseline/lat))`. Reproduce=+0.2,
   2×=+0.8, 4×=+1.4, 8×=+2.0. Deep optimization is now what pays; copying is cheap.

2. **csim overstates correctness.** The reward gate is csim, but csim→cosim collapses (one arm went
   23/26 csim → 5/26 cosim). csim-passing-but-cosim-failing kernels earned positive reward.
   cosim in the RL loop is too slow (~30 min/design) to gate every rollout, so:
   - training keeps the csim gate + degeneracy guard + `__SYNTHESIS__` ban (all necessary), and
   - **every reported number is cosim**, never csim (see §Evaluate).

## Split discipline (the part most easily gotten wrong)

The canonical split is `splits.json`. Freeze it; cite it, not any per-run config.

- **Train GRPO on the `train` split only** (22 polybench benches).
- **Headline generalization on the truly-unseen benches**: `val` (gramschmidt) + `test`
  (durbin, floyd-warshall, gemm, trmm) + **doitgen** (in the eval suite but never in any corpus)
  = 6 clean benchmarks.
- The other 20 of polybench-26 are in training — report them as **in-domain** ("did it learn to
  optimize these"), clearly labeled. **Never headline the full 26 as generalization**: 20/26 would
  be contaminated.
- **Keep a cross-domain eval** (train on rodinia `kernels_c2hls`, test on polybench). A model that
  optimizes a *different benchmark family* than it trained on is the strongest evidence the RL
  taught general HLS skill rather than memorizing. This is GRPO-v1's role.

### The two GRPO runs and what each tests
| | trained on | polybench eval is | tests |
|---|---|---|---|
| **GRPO-v1** (`qwen7b_grpo_base_v1`) | rodinia (17) | **out-of-domain** | cross-domain generalization |
| **GRPO-v2** (job in queue) | polybench train (22) | **in-domain + held-out** | whether same-domain RL beats cross-domain |

GRPO-v1 already beats base and offline-SFT on polybench despite being out-of-domain. GRPO-v2 (with
the corrected reward) tests whether in-domain RL, now actually incentivized to optimize, closes
more of the frontier gap.

## Reward reference

The speedup denominator is each task's `baseline` QoR (`hls_baseline.cpp`, synthesized by
`python -m hlsenv baseline`). It is a *fixed, correct reference*, not the model's own prior output.
Because it is the clean (un-aggressively-optimized) reference, beating it requires genuine
optimization — which is exactly what the log-scale term now rewards. Baselines are
Vitis-version/part specific; regenerate them on any new site.

## Anti-reward-hacking (all load-bearing, keep them)
- **csim gate**: no positive reward without passing the golden testbench.
- **`__SYNTHESIS__` ban**: the canonical exploit is `#ifdef`-ing the kernel body away so csim runs
  the C path while HLS synthesizes nothing.
- **Degeneracy guard**: a csim-passing *empty* circuit (e.g. tile > problem size → zero-iteration
  loop) would otherwise collect a huge fake speedup. Note it still appears at *inference* even when
  penalized in training (a model occasionally emits a degenerate kernel), so the eval parser
  guards against it too.
- **Speedup ratio cap (8×)**: bounds the incentive so one lucky outlier can't dominate.

## Evaluate
- **Report cosim, not csim.** csim systematically overstates correctness here.
- Degeneracy-guard the eval parser (`scripts/arm_tiers.py`, `polybench_matrix.py`).
- Compare on matched benchmark sets, same part/clock, against **gold** and, where available,
  **frontier** arms (`schema_records.jsonl`). Gold is the reference the models should beat; frontier
  shows the achievable ceiling (2–93× past gold).
- The honest headline is the cosim pass count and median cycles **on the held-out set**, with the
  in-domain set reported separately.

## Checklist for a clean claim
1. Train on `splits.json:train`; never eval-headline a benchmark that was in training.
2. Use the log-scale reward (correct reward = optimization pressure, not translation).
3. Regenerate baselines for the target toolchain.
4. Run the §7 smoke test in REPRODUCE.md (esp. `frac_reward_zero_std ≈ 0`).
5. Report cosim on held-out + doitgen; keep a cross-domain row; compare to gold and frontier.
