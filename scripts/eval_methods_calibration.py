"""Calibrate SDK-free proxy metrics against the corpus's REAL Cerebras-SDK labels.

Without the Cerebras SDK we cannot compute real pass/fail or cycles_send for NEW
generations. But the historical corpus was labeled by the SDK when it was available.
This script measures how well each pure-Python proxy predicts the real label, so the
proxy can be trusted on new (open-model) generations.

Design: the DPO corpus gives matched pairs — same prompt, `chosen` (SDK pass) vs
`rejected` (SDK fail). A good proxy should rank chosen above rejected. We report, per
proxy: P(proxy-good | pass), P(proxy-good | fail), and pairwise discrimination
(fraction of pairs where the proxy strictly prefers chosen over rejected).

Proxies (all pure-Python, no cslc/cs_python):
  syntax_valid   benchmark_csl.check_csl_syntax (reference-free)
  code_sim       benchmark_csl.code_similarity to the kernel's gold CSL
  contract_ok    contract_check.validate_against_reference vs the kernel's gold CSL

Gold reference per kernel = the min-cycles_send passing CSL from the SFT corpus.

Usage:
  PYTHONPATH=/eagle/argonne_tpc/lyb/xkernel-v1_0/code_translation \
  python eval_methods_calibration.py <sft_jsonl> <dpo_jsonl> <out_json>
"""
from __future__ import annotations
import json, re, sys, statistics
from collections import defaultdict

from benchmark_csl import code_similarity, check_csl_syntax
import contract_check as cc

_FENCE = re.compile(r"```(?:csl)?\s*\n(.*?)```", re.DOTALL)
def extract_csl(t):
    m = re.search(r"```csl\s*\n(.*?)```", t or "", re.DOTALL)
    if m: return m.group(1).strip()
    m = _FENCE.search(t or "")
    return m.group(1).strip() if m else (t or "").strip()

def content(m):
    c = m.get("content"); return c if isinstance(c, str) else json.dumps(c)

def main(sft, dpo, outp):
    # ---- Pass 1: per-kernel gold reference (min cycles_send passing CSL) ----
    gold = {}  # kernel -> (cycles, csl)
    for line in open(sft):
        try: r = json.loads(line)
        except: continue
        md = r.get("metadata", {})
        if md.get("status") != "pass": continue
        msgs = r.get("messages", [])
        la = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
        if not la: continue
        csl = extract_csl(content(la))
        if "```" not in content(la) and not csl: continue
        cyc = md.get("cycles_send")
        cyc = cyc if isinstance(cyc, (int, float)) else 10**18
        k = md.get("kernel", "?")
        if k not in gold or cyc < gold[k][0]:
            gold[k] = (cyc, csl)
    print(f"gold references built for {len(gold)} kernels", file=sys.stderr)

    # ---- Pass 2: DPO pairwise calibration ----
    # accumulate per proxy: pass-good, fail-good counts; pairwise preference
    agg = defaultdict(lambda: dict(pass_good=0, fail_good=0, n=0,
                                   pair_correct=0, pair_wrong=0, pair_tie=0,
                                   sim_pass=[], sim_fail=[]))
    n_pairs = 0
    for line in open(dpo):
        try: r = json.loads(line)
        except: continue
        k = (r.get("metadata") or {}).get("kernel", "?")
        ref = gold.get(k, (None, None))[1]
        ch = extract_csl(r.get("chosen", ""))
        rj = extract_csl(r.get("rejected", ""))
        if not ch or not rj: continue
        n_pairs += 1

        # syntax_valid (reference-free boolean)
        sv_c = bool(check_csl_syntax(ch).get("syntax_valid"))
        sv_r = bool(check_csl_syntax(rj).get("syntax_valid"))
        a = agg["syntax_valid"]; a["n"] += 1
        a["pass_good"] += sv_c; a["fail_good"] += sv_r
        a["pair_correct"] += (sv_c and not sv_r); a["pair_wrong"] += (sv_r and not sv_c)
        a["pair_tie"] += (sv_c == sv_r)

        if ref:
            # contract_ok boolean
            co_c = cc.validate_against_reference(ch, ref) is None
            co_r = cc.validate_against_reference(rj, ref) is None
            a = agg["contract_ok"]; a["n"] += 1
            a["pass_good"] += co_c; a["fail_good"] += co_r
            a["pair_correct"] += (co_c and not co_r); a["pair_wrong"] += (co_r and not co_c)
            a["pair_tie"] += (co_c == co_r)

            # code_sim continuous — pairwise: chosen more similar to gold?
            sc = code_similarity(ch, ref); sr = code_similarity(rj, ref)
            a = agg["code_sim_vs_gold"]; a["n"] += 1
            a["sim_pass"].append(sc); a["sim_fail"].append(sr)
            a["pair_correct"] += (sc > sr); a["pair_wrong"] += (sr > sc); a["pair_tie"] += (sc == sr)

    out = {"n_dpo_pairs": n_pairs, "n_gold_kernels": len(gold), "proxies": {}}
    for name, a in agg.items():
        n = a["n"] or 1
        row = {
            "n": a["n"],
            "pairwise_discrimination": round(a["pair_correct"] / n, 3),
            "pairwise_wrong": round(a["pair_wrong"] / n, 3),
            "pairwise_tie": round(a["pair_tie"] / n, 3),
            "net_signal": round((a["pair_correct"] - a["pair_wrong"]) / n, 3),
        }
        if a["sim_pass"]:
            row["mean_sim_pass(chosen)"] = round(statistics.mean(a["sim_pass"]), 4)
            row["mean_sim_fail(rejected)"] = round(statistics.mean(a["sim_fail"]), 4)
        else:
            row["rate_good_on_pass(chosen)"] = round(a["pass_good"] / n, 3)
            row["rate_good_on_fail(rejected)"] = round(a["fail_good"] / n, 3)
        out["proxies"][name] = row

    json.dump(out, open(outp, "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
