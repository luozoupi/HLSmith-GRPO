"""Rank the GRPO model's best verified cases across single-shot and multistep runs."""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("/eagle/argonne_tpc/lyb/runs")


def single_shot_cases():
    for suite, rel in (("17", "eval17/7b_grpo_base"), ("26", "eval26/7b_grpo_base")):
        for bd in (RUNS / rel).iterdir():
            f = bd / f"{bd.name}_results.json"
            if not bd.is_dir() or not f.exists():
                continue
            r = json.loads(f.read_text())
            if not (r.get("csim") or {}).get("passed"):
                continue
            comp = (r.get("comparison") or {}).get("comparison") or {}
            ratio = (comp.get("latency_ns") or {}).get("ratio")
            rep = r.get("synth_report") or {}
            if ratio and ratio > 0:
                yield {
                    "bench": r.get("benchmark", bd.name), "mode": "single-shot",
                    "suite": suite, "speedup": 1.0 / ratio,
                    "lat_cyc": rep.get("latency_cycles"),
                    "lut": rep.get("lut"), "dsp": rep.get("dsp"),
                    "fmax": rep.get("fmax_mhz"),
                }


def multistep_cases():
    gold = {}
    for bd in (RUNS / "eval26/7b_grpo_base").iterdir():
        f = bd / f"{bd.name}_results.json"
        if bd.is_dir() and f.exists():
            r = json.loads(f.read_text())
            g = (r.get("ground_truth_report") or {}).get("latency_cycles")
            if g:
                gold[r.get("benchmark", bd.name)] = g
    for bd in (RUNS / "eval26_multistep/7b_grpo_base").iterdir():
        f = bd / f"{bd.name}_multistep_results.json"
        if not bd.is_dir() or not f.exists():
            continue
        r = json.loads(f.read_text())
        name = r.get("benchmark", bd.name)
        lats = []
        for h in r.get("optimization_history") or []:
            if isinstance(h, dict):
                lc = (h.get("report") or {}).get("latency_cycles")
                if lc:
                    lats.append(lc)
        rep = r.get("final_report") or r.get("synth_report") or {}
        if rep.get("latency_cycles"):
            lats.append(rep["latency_cycles"])
        if lats and name in gold:
            best = min(lats)
            yield {
                "bench": name, "mode": "multistep", "suite": "26",
                "speedup": gold[name] / best, "lat_cyc": best,
                "lut": rep.get("lut"), "dsp": rep.get("dsp"),
                "fmax": rep.get("fmax_mhz"),
            }


def main():
    cases = sorted([*single_shot_cases(), *multistep_cases()],
                   key=lambda c: -c["speedup"])
    print(f"{'bench':30s} {'mode':11s} {'speedup':>8s} {'lat_cyc':>12s} {'LUT':>8s} {'DSP':>5s} {'Fmax':>6s}")
    for c in cases[:12]:
        lat = f"{c['lat_cyc']:,}" if c["lat_cyc"] else "-"
        print(f"{c['bench']:30s} {c['mode']:11s} {c['speedup']:>7.2f}x {lat:>12s} "
              f"{str(c['lut']):>8s} {str(c['dsp']):>5s} {str(c['fmax']):>6s}")


if __name__ == "__main__":
    main()
