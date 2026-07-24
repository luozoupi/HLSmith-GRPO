"""Proxy-score held-out CSL generations WITHOUT the Cerebras SDK.

Real correctness/cycles_send needs cslc + cs_python on a Cerebras machine, which
Polaris lacks. This scores each generation with the harness's pure-Python checks,
comparing the fine-tuned model, the base model, and the commercial-model artifact
(the corpus's known-good CSL, used as the reference).

Metrics per generation (vs the commercial artifact for the same prompt):
  gen_rate       produced a non-empty CSL block
  syntax_valid   benchmark_csl.check_csl_syntax -> has CSL idioms, no CUDA leakage
  contract_ok    contract_check.validate_against_reference == None
                 (frozen f_tic/f_toc bodies + frozen params preserved)
  code_sim       benchmark_csl.code_similarity (difflib ratio, 0..1) to the artifact
  exact_match    byte-identical to the artifact CSL (rare)

Also writes an export tarball (generated CSL + manifest) so real cycles_send can be
scored later on a Cerebras-SDK machine (out of Polaris scope).

Usage (needs the origin/v1_0 worktree on PYTHONPATH):
  PYTHONPATH=/eagle/argonne_tpc/lyb/xkernel-v1_0/code_translation \
  python score_csl_proxy.py \
    --gen finetuned=/eagle/argonne_tpc/lyb/runs/eval/gen_csl-sft.jsonl \
    --gen base=/eagle/argonne_tpc/lyb/runs/eval/gen_base.jsonl \
    --out-dir /eagle/argonne_tpc/lyb/runs/eval --export-tar
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import tarfile
from collections import defaultdict
from pathlib import Path

from benchmark_csl import code_similarity, check_csl_syntax  # pure-python, no SDK
import contract_check as cc

_FENCE = re.compile(r"```(?:csl)?\s*\n(.*?)```", re.DOTALL)


def extract_csl(text: str) -> str:
    m = re.search(r"```csl\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = _FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def score_row(row: dict) -> dict:
    gen = (row.get("gen_csl") or "").strip()
    ref = extract_csl(row.get("reference_csl") or "")
    produced = bool(gen) and row.get("status") == "ok"
    if not produced:
        return {"kernel": row["kernel"], "produced": False, "syntax_valid": False,
                "contract_ok": False, "code_sim": 0.0, "exact_match": False}
    syn = check_csl_syntax(gen)
    contract = cc.validate_against_reference(gen, ref) if ref else "no-reference"
    return {
        "kernel": row["kernel"],
        "produced": True,
        "syntax_valid": bool(syn.get("syntax_valid")),
        "contract_ok": contract is None,
        "contract_violation": None if contract is None else str(contract)[:200],
        "code_sim": round(code_similarity(gen, ref), 4) if ref else 0.0,
        "exact_match": gen == ref,
    }


def aggregate(scored: list[dict]) -> dict:
    def agg(rows):
        n = len(rows)
        if n == 0:
            return {"n": 0}
        sims = [r["code_sim"] for r in rows if r["produced"]]
        return {
            "n": n,
            "gen_rate": round(sum(r["produced"] for r in rows) / n, 3),
            "syntax_valid_rate": round(sum(r["syntax_valid"] for r in rows) / n, 3),
            "contract_ok_rate": round(sum(r["contract_ok"] for r in rows) / n, 3),
            "code_sim_mean": round(st.mean(sims), 4) if sims else 0.0,
            "code_sim_median": round(st.median(sims), 4) if sims else 0.0,
            "exact_match_rate": round(sum(r["exact_match"] for r in rows) / n, 3),
        }
    by_kernel = defaultdict(list)
    for r in scored:
        by_kernel[r["kernel"]].append(r)
    return {"overall": agg(scored),
            "per_kernel": {k: agg(v) for k, v in sorted(by_kernel.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="append", required=True,
                    help="label=path.jsonl (repeatable), e.g. finetuned=gen_csl-sft.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--export-tar", action="store_true",
                    help="stage generated CSL + manifest for later real-QoR scoring")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scoreboard = {}
    export_root = out / "export_for_real_qor"

    for spec in args.gen:
        label, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path)]
        scored = [score_row(r) for r in rows]
        scoreboard[label] = aggregate(scored)
        # per-row detail
        with (out / f"scored_{label}.jsonl").open("w") as g:
            for r, s in zip(rows, scored):
                g.write(json.dumps({**s, "commercial_model": r.get("commercial_model"),
                                    "ref_cycles": r.get("ref_cycles")}) + "\n")
        if args.export_tar:
            for i, r in enumerate(rows):
                if not (r.get("gen_csl") and r.get("status") == "ok"):
                    continue
                d = export_root / label / r["kernel"] / f"{i:04d}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "generated.csl").write_text(r["gen_csl"])
                (d / "manifest.json").write_text(json.dumps({
                    "kernel": r["kernel"], "label": label, "src_index": r.get("src_index"),
                    "served_model": r.get("served_model"),
                    "commercial_model": r.get("commercial_model"),
                    "ref_cycles": r.get("ref_cycles"), "run_dir": r.get("run_dir"),
                }, indent=2))

    with (out / "scoreboard.json").open("w") as g:
        json.dump(scoreboard, g, indent=2)

    # printed table
    cols = ["n", "gen_rate", "syntax_valid_rate", "contract_ok_rate",
            "code_sim_mean", "code_sim_median", "exact_match_rate"]
    print(f"\n{'label':12s} " + " ".join(f"{c:>17s}" for c in cols))
    for label, agg in scoreboard.items():
        o = agg["overall"]
        print(f"{label:12s} " + " ".join(f"{str(o.get(c,'')):>17s}" for c in cols))
    print("\nPer-kernel detail + full scoreboard -> ", out / "scoreboard.json")
    if args.export_tar:
        tarpath = out / "csl_generations_for_real_qor.tar.gz"
        with tarfile.open(tarpath, "w:gz") as tf:
            tf.add(export_root, arcname="export_for_real_qor")
        print("Real-QoR export bundle ->", tarpath)


if __name__ == "__main__":
    main()
