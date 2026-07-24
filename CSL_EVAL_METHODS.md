# Evaluating the CUDA→CSL agent with open models — without the Cerebras SDK

**Problem.** The ground-truth quality signal for a generated CSL kernel is:
compile with `cslc` → run on the WSE simulator with `cs_python` → get `status`
(pass/fail) and `cycles_send` (QoR). That toolchain **is not available on Polaris**
(no Cerebras SDK/chip). So for *new* generations from open models we cannot compute
the real label directly. This documents SDK-free ways to evaluate generation quality,
how trustworthy each is, and the measured results.

## 1. Method families (all runnable on Polaris)

| # | Method | Needs SDK? | Measures | Strength / limit |
|---|--------|-----------|----------|------------------|
| 1 | Reference-based structural (code-similarity, exact match, symbol/import overlap) | no | closeness to a known-good CSL | cheap; low absolute values, weak correctness link |
| 2 | Reference-free static validators (CSL syntax, launch-contract `f_tic`/`f_toc`+frozen params, builtin whitelist, layout/mesh consistency) | no | surface well-formedness + contract compliance | cheap filter; **necessary not sufficient** |
| 3 | **Proxy calibration vs historical SDK labels** | no (uses stored labels) | how well a proxy predicts real pass/fail | quantifies proxy trust — see §3 |
| 4 | LLM-as-judge (strong open model scores CSL on a rubric) | no | semantic correctness estimate | best SDK-free correctness proxy; needs a served judge |
| 5 | Self-consistency / proxy-pass@k (N samples, agreement) | no | stability / confidence | catches flakiness; not absolute |
| 6 | Relative model ranking on held-out kernels | no | base vs fine-tuned vs commercial | gross differences are meaningful |
| 7 | Deferred real verification (export → `cslc`/`cs_python` on a Cerebras host) | **yes (elsewhere)** | true `status` + `cycles_send` | the bridge to ground truth |

The proxy suite we implemented (pure-Python, imports without the SDK) — from the
`origin/v1_0` harness: `benchmark_csl.code_similarity`, `benchmark_csl.check_csl_syntax`,
`contract_check.validate_against_reference`, plus `bench_structure`/`mesh_gate`.

## 2. Held-out model comparison (relative ranking, method #1/#2/#6)

Generate CSL for kernels **held out of training** via a local vLLM server, score each
generation against the commercial-model artifact for the same prompt. Fine-tuned vs base
tells you whether SFT helped; commercial artifacts are the ceiling.

**Result — Qwen3-8B (32K LoRA, checkpoint @ epoch 2.55), 6 held-out kernels, 24 gens/model:**

| metric | fine-tuned | base | commercial artifact |
|---|--:|--:|--:|
| gen_rate (emits a CSL block) | **1.000** | 0.958 | 1.00 (ref) |
| syntax_valid_rate | 0.750 | **0.875** | — |
| contract_ok_rate | 0.417 | 0.417 | — |
| code_sim_mean (to artifact) | 0.156 | **0.172** | 1.00 (self) |

**Sober, honest read:** on a strong 8B base, 2.55-epoch LoRA moved the static proxies *within
noise* on held-out kernels — base is even marginally ahead on syntax/similarity. The one clean gain
is **gen_rate** (fine-tuned always returns a well-formed CSL block; base occasionally doesn't).
This agrees with §3: the proxies are weak, so small deltas aren't real-pass-rate evidence — and the
method correctly declines to manufacture a win. Caveats: small sample (24 gens), held-out = the
hardest kernels, completions capped at 4,096 tokens (can clip long kernels). *(A pipeline smoke on
the weak Qwen2.5-Coder-1.5B base did show a large SFT gain — 0.833 vs 0.583 syntax — as expected
when the base is weak; that gain shrinks against a strong base.)*

## 3. Proxy calibration against REAL SDK labels (method #3 — the key result)

The corpus still carries SDK labels from when the chip was available. The **DPO** split
gives matched pairs: same prompt, `chosen` = SDK **pass**, `rejected` = SDK **fail**. A
trustworthy proxy should rank chosen above rejected. Measured over **2,247 pairs** (gold
reference per kernel = min-`cycles_send` passing CSL):

| proxy | good-rate on PASS (chosen) | good-rate on FAIL (rejected) | prefers chosen | wrong | strength (0=chance,1=oracle) |
|---|--:|--:|--:|--:|--:|
| **LLM-judge** (Qwen2.5-Coder-32B, n=150) | — | — | 60.0% | 40.0% | **+0.20** |
| `syntax_valid` (ref-free) | 95.2% | 87.9% | 11.7% | 4.5% | +0.073 |
| `code_sim` → gold | mean 0.136 | mean 0.127 | 51.3% | 48.2% | +0.031 |
| `contract_ok` (vs gold) | 70.8% | 69.3% | 3.6% | 2.1% | +0.015 |

**Interpretation (important, and honest):** the static proxies are *weak* predictors of real
pass/fail — failing CSL is **mostly still syntactically well-formed and contract-compliant**
(88% of rejected pass the syntax check; 69% pass the contract check); it fails for deeper reasons
(compile errors, semantic/timing bugs) that pure-Python cannot see. The **LLM-judge is ~3× stronger**
(strength 0.20 vs ≤0.073) — a strong 32B *code* model, shown the CUDA source + both candidates,
picks the real passing one **60%** of the time (chance = 50%). But it is still only weak-moderate,
and shows a **strong position bias** (picked the 2nd candidate 84% of the time) and wild per-kernel
variance (Cholesky/Jacobi/Residual = 1.0; GEMV-Collectives/PDFT/Single-Tile-Matvec = 0.0). So:

- The static proxies are legitimate **cheap filters** (they *do* catch CUDA leakage,
  missing CSL idioms, broken `f_tic`/`f_toc`) and their signal is consistently
  net-positive — but they are **not correctness oracles**. Improvements on them are
  *directional evidence*, not proof of higher real pass-rate.
- This is why the recommended pipeline is a **ladder**, not a single proxy.

## 4. Recommended evaluation ladder

1. **Static filter (free, instant):** drop generations failing `check_csl_syntax` /
   `contract_check`. Removes obviously-broken output before spending anything else.
2. **LLM-as-judge (free on Polaris, GPU):** a strong open model scores the surviving CSL
   for semantic correctness vs the CUDA source + contract. Calibrate the judge the same
   way (pairwise DPO accuracy) to report its trust — expected to beat the static proxies.
3. **Relative ranking (free):** report the §2 scoreboard (fine-tuned vs base vs commercial)
   on held-out kernels; use gross gaps, not decimals.
4. **Deferred ground truth:** export the top candidates (`score_csl_proxy.py --export-tar`)
   and run real `cslc`/`cs_python` on a Cerebras host for the true `status`/`cycles_send`.

## 5. Reproduce

```bash
# proxy calibration vs real SDK labels (this report's §3 table)
PYTHONPATH=/eagle/argonne_tpc/lyb/xkernel-v1_0/code_translation \
python scripts/eval_methods_calibration.py \
  /eagle/argonne_tpc/lyb/data/csl_sft/raw/corpus/sft_corpus.jsonl \
  /eagle/argonne_tpc/lyb/data/csl_sft/raw/corpus/dpo_corpus.jsonl \
  /eagle/argonne_tpc/lyb/runs/eval_methods/calibration.json

# held-out model scoreboard (§2): serve + generate + proxy-score
qsub scripts/pbs/csl_eval.pbs   # see CSL_SFT_README.md for -v args
```
Artifacts: `runs/eval_methods/calibration.json`, `runs/eval_qwen3_8b/scoreboard.json`,
`runs/*/csl_generations_for_real_qor.tar.gz` (deferred-verification bundle).
