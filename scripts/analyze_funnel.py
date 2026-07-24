"""Bottleneck funnel analysis over c2hls results.

For each benchmark: where did the run die / how much did each agent role
contribute? Stages: gold-ref valid -> translate (first-shot synth) ->
repair recovery -> csim -> quality repair -> QoR vs GT.

Usage: python analyze_funnel.py <results_root> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def analyze_root(root: Path):
    rows = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        f = bench_dir / f"{bench_dir.name}_results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        th = r.get("turn_history") or []
        b_turns = [t for t in th if t.get("phase") == "B"]
        first_shot = bool(b_turns and b_turns[0].get("success"))
        synth_ok = any(t.get("success") for t in b_turns)
        recovered = synth_ok and not first_shot
        csim = (r.get("csim") or {})
        qr = r.get("quality_repair") or {}
        comp = (r.get("comparison") or {}).get("comparison") or {}
        lat = (comp.get("latency_ns") or {}).get("ratio")
        rows.append({
            "bench": r.get("benchmark", bench_dir.name),
            "gold_valid": r.get("ground_truth_status") == "valid",
            "b_turns_used": len(b_turns),
            "first_shot_synth": first_shot,
            "synth_ok": synth_ok,
            "repair_recovered": recovered,
            "csim_pass": bool(csim.get("passed")),
            "csim_ran": bool(csim.get("ran")),
            "qr_attempted": bool(qr.get("attempted")),
            "qr_applied": bool(qr.get("applied")),
            "lat_ratio": lat,
            "success": bool(r.get("success")),
        })
    return rows


def report(rows):
    n = len(rows)
    if not n:
        print("  (no results)")
        return
    gold = [r for r in rows if r["gold_valid"]]
    synth = [r for r in gold if r["synth_ok"]]
    first = [r for r in gold if r["first_shot_synth"]]
    rec = [r for r in gold if r["repair_recovered"]]
    unrecovered = [r for r in gold if not r["synth_ok"]]
    csim_p = [r for r in synth if r["csim_pass"]]
    ratios = sorted(r["lat_ratio"] for r in csim_p if r["lat_ratio"])
    print(f"  benchmarks analyzed:        {n}")
    print(f"  gold reference valid:       {len(gold)}/{n}"
          f"   <- lost to benchmark data, not agents")
    print(f"  TRANSLATOR first-shot synth:{len(first)}/{len(gold)}")
    print(f"  REPAIR recovered (of {len(gold)-len(first)} fails): {len(rec)}"
          f"   unrecovered: {len(unrecovered)}")
    print(f"  net synth rate:             {len(synth)}/{len(gold)}")
    print(f"  CSIM pass (of synthed):     {len(csim_p)}/{len(synth)}"
          f"   <- correctness bottleneck if low")
    qr_a = [r for r in synth if r["qr_attempted"]]
    qr_ok = [r for r in synth if r["qr_applied"]]
    print(f"  QUALITY repair attempted/applied: {len(qr_a)}/{len(qr_ok)}")
    if ratios:
        med = ratios[len(ratios) // 2]
        wins = sum(1 for x in ratios if x <= 1.0)
        print(f"  latency ratio vs GT (csim-passed): med={med:.2f}, "
              f"beats-GT={wins}/{len(ratios)}")
    print(f"  end-to-end success:         {sum(r['success'] for r in rows)}/{n}")
    print("\n  per-benchmark failure attribution:")
    for r in rows:
        if r["success"] and r["csim_pass"]:
            continue
        if not r["gold_valid"]:
            why = "invalid gold reference (benchmark data)"
        elif r["b_turns_used"] == 0:
            why = "TRANSLATOR aborted pre-synth (no code fence or exception)"
        elif not r["synth_ok"]:
            why = f"REPAIR failed to recover ({r['b_turns_used']} turns burned)"
        elif not r["csim_pass"]:
            why = "csim fail (functional correctness — translator semantics)"
        else:
            why = "other"
        print(f"    {r['bench']:18s} {why}")


if __name__ == "__main__":
    for root in map(Path, sys.argv[1:]):
        print(f"\n===== {root} =====")
        report(analyze_root(root))
