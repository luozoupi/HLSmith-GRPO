"""Detailed per-benchmark arm report: status, speedup over gold reference, resources.

Speedup = GT latency_ns / generated latency_ns (>1 = faster than gold).
Usage: python make_arm_report.py <eval_root_with_arm_subdirs>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = ["7b_base", "7b_sft", "7b_sft_grpo", "7b_grpo_base"]


def load_arm(root: Path, arm: str) -> dict:
    out = {}
    d = root / arm
    if not d.is_dir():
        return out
    for bd in sorted(p for p in d.iterdir() if p.is_dir()):
        f = bd / f"{bd.name}_results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        comp = (r.get("comparison") or {}).get("comparison") or {}
        lat = comp.get("latency_ns") or comp.get("latency_cycles") or {}
        ratio = lat.get("ratio")
        rep = r.get("synth_report") or {}
        out[r.get("benchmark", bd.name)] = {
            "synth": bool(rep),
            "csim": bool((r.get("csim") or {}).get("passed")),
            "gold_bad": r.get("ground_truth_status") != "valid",
            "speedup": (1.0 / ratio) if ratio else None,
            "lut": rep.get("lut"), "ff": rep.get("ff"),
            "dsp": rep.get("dsp"), "bram": rep.get("bram"),
            "fmax": rep.get("fmax_mhz"),
            "lut_ratio": (comp.get("lut") or {}).get("ratio"),
            "dsp_ratio": (comp.get("dsp") or {}).get("ratio"),
        }
    return out


def cell(e):
    if e is None:
        return "-"
    if e["gold_bad"]:
        return "no-GT"
    if not e["synth"]:
        return "FAIL"
    if not e["csim"]:
        return "synth,csim-X"
    return f"{e['speedup']:.2f}x" if e["speedup"] else "pass"


def main(root: str):
    root = Path(root)
    arms = {a: load_arm(root, a) for a in ARMS}
    benches = sorted({b for d in arms.values() for b in d})

    print("| benchmark | " + " | ".join(ARMS) + " |")
    print("|---|" + "---|" * len(ARMS))
    for b in benches:
        row = [cell(arms[a].get(b)) for a in ARMS]
        print(f"| {b} | " + " | ".join(row) + " |")

    print("\n### Resources of csim-passed designs (winner arm vs others)")
    print("| benchmark | arm | speedup | LUT | FF | DSP | BRAM | Fmax MHz | LUT vs GT | DSP vs GT |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b in benches:
        for a in ARMS:
            e = arms[a].get(b)
            if e and e["csim"] and e["speedup"]:
                print(f"| {b} | {a} | {e['speedup']:.2f}x | {e['lut']} | {e['ff']} | "
                      f"{e['dsp']} | {e['bram']} | {e['fmax']} | "
                      f"{e['lut_ratio']:.2f} | " if e['lut_ratio'] else "- | ", end="")
                print(f"{e['dsp_ratio']:.2f} |" if e["dsp_ratio"] else "- |")


if __name__ == "__main__":
    main(sys.argv[1])
