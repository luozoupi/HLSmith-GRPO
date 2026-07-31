# GRPO for HLS kernel generation — algorithm and reward design

Technical specification of the training method. Companions: `REPRODUCE.md` (how to run it),
`METHODOLOGY.md` (how to make a defensible claim from it), `splits.json` (data partition).

---

## 1. Problem formulation

We treat HLS kernel generation as a **single-step contextual bandit**, not a multi-step MDP:

- **Context (state)** `s`: a task prompt — plain C/C++ source, the benchmark header, and a
  requirement line naming the top-level function. One context per benchmark, from `spec.md`.
- **Action** `a`: one complete model generation (a fenced C++ block, ≤ `max_completion_length`
  tokens). The action space is the token space; the *effective* action is the extracted kernel.
- **Reward** `r(s,a) ∈ [-1, 2]`: obtained by **actually compiling, simulating and synthesizing the
  generated kernel with Vitis HLS** and scoring correctness + quality-of-results (§4).
- **Episode**: one generation. No environment transitions, no bootstrapping, no value function.

This matters: the reward is a *program-analysis oracle*, not a learned reward model or a
heuristic. Every reward is grounded in a real toolchain run at a fixed target
(`xcu280-fsvh2892-2L-e @ 3.33 ns`). The cost is that each reward evaluation takes seconds to
minutes of CPU, which dominates wall-clock (§6).

## 2. Algorithm: GRPO (Group Relative Policy Optimization)

GRPO replaces PPO's learned critic with a **group-relative baseline**: sample G completions per
prompt and use the group's own reward statistics to compute advantages. No value network is
trained, which suits our setting — a critic would have to predict Vitis QoR from tokens, which is
harder than the policy task itself.

For prompt `s`, sample a group `{a_1 … a_G}` from the current policy `π_θ`, score each to get
`{r_1 … r_G}`, and form the advantage

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)          # scale_rewards = "group"
```

Then optimize a clipped importance-weighted objective (token-level ratios):

```
ρ_i,t = π_θ(a_i,t | s, a_i,<t) / π_θ_old(a_i,t | s, a_i,<t)
L = -E[ min( ρ_i,t · A_i , clip(ρ_i,t, 1-ε, 1+ε) · A_i ) ] + β · KL(π_θ ‖ π_ref)
```

### Configured hyperparameters (TRL 1.8 `GRPOConfig`)

| Parameter | Value | Notes |
|---|---|---|
| `num_generations` (G) | **8** | group size; the baseline is the mean of these 8 |
| `loss_type` | **dapo** | token-level loss normalization (DAPO) rather than per-sequence mean |
| `beta` (β, KL coef) | **0.0** | **no KL penalty — see below** |
| `epsilon` (ε) | 0.2 | PPO-style clip range |
| `scale_rewards` | group | advantages divided by within-group std |
| `num_iterations` | 1 | one optimization pass per batch → **strictly on-policy** (ρ ≈ 1) |
| `importance_sampling_level` | token | per-token ratios |
| `temperature` / `top_p` / `top_k` | 1.0 / 1.0 / 0 | unbiased sampling — full diversity for the group baseline |
| `max_completion_length` | 1536 | 2048 caused late-run OOM; median completion ≈ 509 tok |
| `per_device_train_batch_size` | 1 | |
| `gradient_accumulation_steps` | 16 | |
| effective global batch | 3 × 1 × 16 = **48** | = 6 groups of 8 (must be divisible by G) |
| `learning_rate` | **1e-5** | tuned for LoRA; **script default 1e-6 is 10× too low** |
| optimizer / precision | AdamW, **bf16**, gradient checkpointing | |
| `max_steps` | 100 | |
| `save_steps` | 2 | frequent — the queue is preemptible (§6) |

**β = 0 is a deliberate consequence, not an oversight.** With no KL term, no reference-policy copy
is held in GPU memory (a material saving at 7B on 40 GB cards) and the policy is free to move
toward high-QoR regions. The safeguard against degeneration is not KL but the **reward ladder
itself**: any drift that breaks compilation or csim is punished at −0.6 to −1.0, which bounds
policy collapse far more directly than a KL leash. The cost is no formal guarantee of staying near
the base model — acceptable at 100 steps with LoRA (§5), and we monitor for collapse via
`frac_reward_zero_std`.

### Dataset construction
`build_dataset()` emits one conversational row per task, repeated `--repeats` (default 64) times,
so a task recurs across steps and the policy sees it under an evolving policy. Tasks are dropped if
they have **no baseline QoR** (the reward denominator is undefined) or if `timeout_s > 900`
(slow syntheses stall the rollout barrier). Of 17 rodinia tasks, **14** survive; of 22 polybench
train tasks, 22 survive.

## 3. Rollout → reward pipeline

Per completion:

1. **Extract** (`hlsenv/extract.py`) — take the last ```cpp/```c++ fenced block; fall back to the
   last C++-looking fence, then any fence, then raw text if it looks like a function definition.
   Nothing extractable → `NO_CODE`.
2. **Screen for reward hacking** — reject if the source contains a banned token (§4.3).
3. **Synthesize** (`hlsenv/runner.py`) in an isolated scratch dir on node-local NVMe:
   ```tcl
   set_param general.maxThreads 1      # 1 thread/worker: we parallelize across designs
   add_files kernel.cpp ; add_files -tb tb.cpp
   set_part <part> ; create_clock -period <clk>
   csim_design                          # correctness gate vs golden testbench
   csynth_design                        # QoR
   ```
   Run under `subprocess` with `start_new_session=True` so a timeout kills the **whole process
   group** (`vitis_hls` spawns children that otherwise survive and leak).
4. **Classify** the log into `OK / COMPILE_FAIL / CSIM_FAIL / SYNTH_FAIL / TIMEOUT`.
5. **Parse QoR** from `csynth.xml` → latency (best/avg/worst), II, LUT/FF/DSP/BRAM/URAM, estimated
   clock.
6. **Score** (§4). Reports are rsync'd to an archive; scratch is deleted.

Robustness: any unexpected exception in one episode is caught and scored as `SYNTH_FAIL` — a
single bad episode (disk full, parse oddity) must never kill a multi-hour job. Genuine
configuration errors (`vitis_hls` not on PATH) are re-raised to fail fast.

## 4. Reward design

### 4.1 Ladder (terminal, non-OK outcomes)

| Outcome | Reward | Rationale |
|---|---|---|
| `NO_CODE` — no extractable C++ | **−1.0** | worst: not even an attempt |
| `BANNED` — uses an escape hatch (§4.3) | **−1.0** | equal to no-code: never profitable |
| `COMPILE_FAIL` / `SYNTH_FAIL` / `TIMEOUT` | **−0.8** | code exists but is not synthesizable |
| `CSIM_FAIL` — wrong results vs golden TB | **−0.6** | synthesizes but is functionally wrong |
| `DEGENERATE` — csim-passing empty circuit | **−0.6** | scored as if wrong (§4.3) |
| `OK` | **[−0.4, 2.0]**, shaped (§4.2) | |

The ordering encodes the engineering hierarchy *produce code → make it compile → make it correct →
make it fast*. The OK-floor (−0.4) sits **above** `CSIM_FAIL` (−0.6) so a correct-but-slow or
over-budget design is never ranked below a wrong one — the single most important invariant in the
ladder.

### 4.2 Shaped QoR reward (correct designs only)

```
reward = 0.2                                        # base for being correct & synthesizable
       + 0.6 · log2( min(8, baseline_lat / lat) )   # speedup, LOG scale
       − 0.5 · Σ_r max(0, used_r/budget_r − 1)      # resource overage, r ∈ {LUT,FF,DSP,BRAM}
reward = clip(reward, −0.4, 2.0)
```

- **`baseline_lat`** is each task's own reference QoR, produced by synthesizing the reference
  kernel (`python -m hlsenv baseline`). It is a *fixed, correct, per-task* denominator — not the
  model's previous output — so reward is comparable across steps and cannot drift.
- **Budgets** are 50 % of the U280 (LUT 651,840 · FF 1,303,680 · DSP 4,512 · BRAM 2,016). Overage
  is penalized linearly and only beyond 100 % of budget: area is free until it is scarce.
- **Undefined latency** (unbounded loops → `undef`) returns 0.0: synthesized, but QoR is unusable.

**Why log-scale (the key design decision).** The original term was linear and capped at 2×:
`0.6·min(2, base/lat)`. That paid **0.8** — over half the achievable range — for a kernel that
merely *reproduced* the reference, and only +0.6 more for a full 2× speedup. Empirically the model
learned exactly what that reward asked: it translated faithfully to ~1.0× the reference and stopped,
while frontier models achieved 2–93× on the same kernels. The log2 form makes reward grow **+0.6 per
doubling**:

| speedup | 1× (copy) | 2× | 4× | 8× (cap) |
|---|---|---|---|---|
| reward | **+0.2** | +0.8 | +1.4 | **+2.0** |

Copying is now cheap and deep optimization pays across the whole measured headroom (~8–90×). The
ratio is capped at 8× so a single outlier cannot dominate a group's advantage.

### 4.3 Anti-reward-hacking (all three are load-bearing)

A csim-gated QoR reward has three known exploits; each has a specific guard.

1. **`__SYNTHESIS__` ban.** `#ifndef __SYNTHESIS__` lets code behave correctly under C simulation
   while synthesizing to *different* (e.g. empty) hardware — the canonical way to get correctness
   credit and a fake speedup simultaneously. Any occurrence scores −1.0.
2. **Degeneracy check** (`hlsenv/qor.py:is_degenerate`). A do-nothing circuit can vacuously pass a
   weak testbench and post an enormous "speedup". Flagged when: latency ≤ 0; **or** LUT ≤ 100;
   **or** DSP = 0 ∧ LUT < 512 ∧ FF < 128; **or** latency < 2 % of a DSP-using baseline while DSP
   collapsed to 0. Deliberately *not* flagged: legitimately DSP-free kernels with real logic (a
   constant-coefficient stencil synthesizes multiplies into LUTs — jacobi-1d uses 77 k LUT, 0 DSP).
   Real occurrence: a tile size exceeding the problem dimensions makes the compute loop execute
   zero times.
3. **Speedup cap (8×)** — bounds the payoff of any residual exploit.

The csim gate underpins all of it: **no positive reward without passing the golden testbench.**

### 4.4 Known limitation of the gate
csim is a *C-level* check. We have measured that csim systematically overstates correctness
(one evaluation arm: 23/26 csim → 5/26 cosim), so kernels that are correct in C but wrong in RTL
can still earn positive reward. cosim (~30 min/design) is far too slow to gate every rollout, so the
compromise is: **train on csim, report on cosim**. Closing this gap — e.g. periodic cosim audits of
top-k candidates — is the main outstanding improvement to the reward.

## 5. Model and parameterization

**LoRA** (full fine-tuning of 7B does not fit 4×40 GB with a colocated inference engine):

| | |
|---|---|
| rank `r` | 32 |
| `lora_alpha` | 64 (α/r = 2) |
| dropout | 0.05 |
| target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| task type | CAUSAL_LM |

Attention *and* MLP projections are adapted; restricting to attention alone underfits the
pragma-placement behaviour we want to learn. The adapter is merged into the base weights for
evaluation (`scripts/merge_adapter.py`, on a compute node — a login-node cgroup OOM-kills the
~30 GB CPU merge).

## 6. Systems design (why the wall-clock looks the way it does)

**The reward is the bottleneck, not the GPU.** Each rollout requires a full Vitis csim + csynth
(seconds to 10 min). GPUs idle unless synthesis throughput ≥ rollout demand.

- **vLLM in `server` mode**: one GPU runs `trl vllm-serve`; the remaining ranks train. Colocate mode
  (an engine *inside* every trainer rank) does not fit 7B on 40 GB cards — it OOMs in cumem
  sleep-mode, and the custom all-reduce kernel crashes when a TP engine initializes inside a rank
  already holding a CUDA context (patched via `_force_nccl_all_reduce`).
- **CPU/GPU core split**: cores 0–11 serve the trainer ranks and the inference engine; cores 12–31
  run 16 single-threaded `vitis_hls` workers (`set_param general.maxThreads 1` — we parallelize
  *across designs*, not within one). Measured ≈ 240–900 syntheses/hour on 20 cores.
- **Per-rank pool partitioning**: TRL calls the reward function on **every** rank with that rank's
  slice of completions. Without splitting, N ranks × N workers oversubscribe the same cores, so
  `partition_for_rank()` divides both the worker count and the core list by world size.
- **Node-local scratch only.** Synthesis creates thousands of small files; running it on a shared
  parallel filesystem thrashes it. Reports are rsync'd out on exit because node-local NVMe is wiped
  at job end.

**Observed cost**: ~10 min/step from an SFT warmstart (6 h 28 m / 100 steps) vs ~20 min/step from
base (16 h 44 m / 100 steps). The difference is entirely reward-side: a base model emits more
non-compiling code, so more rollouts burn the full synthesis timeout. **Budget walltime from hours,
not steps.**

**Preemption**: on a preemptible queue, `save_steps` must be small enough that a checkpoint lands
before a kill. At 20 min/step, `save_steps=10` needs 200 min and banked *nothing* across several
preemptions; `save_steps=2` (~40 min) survives. Resume is automatic via
`get_last_checkpoint(output_dir)`.

## 7. Diagnostics

| Signal | Healthy | Meaning |
|---|---|---|
| `reward` | rising | e.g. −0.45 → −0.18 over 100 steps (v1); −0.65 → −0.33 over 3 steps (v2, log-reward) |
| `reward_std` | > 0, ideally rising | the reward **discriminates** designs; rising std after the log-reward change confirms the fix |
| `frac_reward_zero_std` | → 0 | fraction of groups where all G completions score identically. **If this pins at 1.0 the run is worthless** — zero advantage, zero gradient, regardless of how long it trains. |
| completion length | ≪ cap | median ≈ 509 tok vs cap 1536, so truncation rarely binds |

`frac_reward_zero_std` is the single most important early check: with G = 8 and a coarse reward
ladder, a policy that fails identically on every sample produces no learning signal at all.

## 8. Design rationale and known limitations

**Why GRPO rather than PPO or DPO.** No critic is needed (a value head predicting Vitis QoR from
tokens is harder than the policy task). No preference pairs are needed — the toolchain provides a
*cardinal* reward, which DPO would discard by binarizing.

**Why single-step.** The agentic framework (translate → repair → optimize) is multi-turn, but we
train the **translator** role in isolation. An ablation showed GRPO's benefit is concentrated
there: routing only the translator to the GRPO adapter gives 14/17 with first-shot synthesis rising
42 % → 100 %. A step-level GRPO variant (`kernels_c2hls_steps`) trained on individual
transformations kept a flat reward curve — step-optimization is a harder RL target.

**Limitations.**
1. **csim-only gate** (§4.4) — the reward cannot see RTL-level correctness.
2. **Latency-dominant objective** — area enters only as an overage penalty, so the policy prefers
   spending area for speed. Observed directly: gesummv traded 10× LUT for 13× speedup.
3. **No legality checking.** The model applies `UNROLL`/`ARRAY_PARTITION` without verifying factors
   divide trip counts or that addressing stays in bounds; the reward punishes the result but gives
   no structured signal about *why*, so out-of-bounds patterns persist.
4. **Fixed baseline denominator** — reward is relative to a per-task reference, so tasks whose
   reference is already near-optimal offer little headroom, while a weak reference (e.g. knn's
   over-engineered double-buffered GT) is easy to beat for reasons unrelated to skill.
5. **β = 0** gives no formal proximity guarantee to the base policy.
