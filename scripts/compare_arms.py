"""Paper-ready comparison: open 7B (GRPO) arms vs commercial (schema_records)
on polybench-26, at synth/csim/cosim tiers, with per-arm and per-benchmark-
intersection QoR.

Multistep handling: the reported design per tier is the FASTEST step that
actually PASSES that tier (multistep's final_report may be a faster design that
failed cosim — reporting it would overstate verified QoR).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, "/eagle/argonne_tpc/lyb/projects/llm-hls")
from hlsenv.qor import is_degenerate


def _deg(lat, dsp, lut, ff=None):
    """Degeneracy check on raw resource numbers (empty/do-nothing circuits)."""
    q = {"lat_worst": lat, "dsp": dsp, "lut": lut, "ff": ff}
    return is_degenerate(q)


def _norm(b):
    return b.replace("hlsfactory_", "").replace("-", "_")


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_single(root):
    """{bench_norm: {synth,csim,cosim(bool), cyc_by_tier}} for single-shot."""
    out = {}
    p = Path(root)
    if not p.exists():
        return out
    for bd in sorted(x for x in p.iterdir() if x.is_dir()):
        f = bd / f"{bd.name}_results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        sr = d.get("synth_report") or {}
        lat = sr.get("latency_cycles")
        deg = _deg(lat, _i(sr.get("dsp")), _i(sr.get("lut")), _i(sr.get("ff")))
        csim = bool((d.get("csim") or {}).get("passed")) and not deg
        cosim = bool((d.get("cosim") or {}).get("passed")) and not deg
        cocyc = (d.get("cosim") or {}).get("cosim_cycles") or lat
        out[_norm(bd.name)] = {
            "synth": bool(d.get("success") or lat), "degenerate": deg,
            "csim": csim, "cosim": cosim,
            "cyc_synth": lat, "cyc_csim": lat if csim else None,
            "cyc_cosim": int(cocyc) if (cosim and cocyc) else None,
        }
    return out


def parse_multistep(root):
    """Per tier, take the fastest STEP that passes that tier."""
    out = {}
    p = Path(root)
    if not p.exists():
        return out
    for bd in sorted(x for x in p.iterdir() if x.is_dir()):
        f = bd / f"{bd.name}_multistep_results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        steps = d.get("optimization_history") or d.get("steps") or []
        best = {"synth": None, "csim": None, "cosim": None}
        any_ok = False
        for s in steps:
            rep = s.get("report") or {}
            lat = rep.get("latency_cycles")
            if not (s.get("success") and lat):
                continue
            lat = int(lat)
            # skip empty/do-nothing steps (reward-hacked "optimizations")
            if _deg(lat, _i(rep.get("dsp")), _i(rep.get("lut")), _i(rep.get("ff"))):
                continue
            any_ok = True
            if best["synth"] is None or lat < best["synth"]:
                best["synth"] = lat
            if (s.get("csim") or {}).get("passed") and (best["csim"] is None or lat < best["csim"]):
                best["csim"] = lat
            if (s.get("cosim") or {}).get("passed") and (best["cosim"] is None or lat < best["cosim"]):
                best["cosim"] = lat
        out[_norm(bd.name)] = {
            "synth": any_ok, "csim": best["csim"] is not None, "cosim": best["cosim"] is not None,
            "cyc_synth": best["synth"], "cyc_csim": best["csim"], "cyc_cosim": best["cosim"],
        }
    return out


def parse_commercial(schema_path):
    """schema_records: per (arm,bench) collect synth/csim(sw_run)/cosim(rtl_sim) + cosim cycles."""
    recs = [json.loads(l) for l in open(schema_path)]
    arms = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        arm = r["implementation"].get("origin_version") or "unlabeled"
        b = _norm("/".join(r["problem"]["group_path"]))
        t = r.get("report_type")
        cell = arms[arm][b]
        if t == "hls_synth" and (r.get("hls_synth") or {}).get("status") == "pass":
            cell["synth"] = True
            pe = (r["hls_synth"].get("PerformanceEstimates") or {}).get("SummaryOfOverallLatency") or {}
            wl = pe.get("Worst-caseLatency")
            if wl not in (None, "undef"):
                cell["cyc_synth"] = int(wl)
            res = (r["hls_synth"].get("AreaEstimates") or {}).get("Resources") or {}
            cell["_res"] = (_i(res.get("DSP") or res.get("DSP48E")), _i(res.get("LUT")), _i(res.get("FF")))
        if t == "sw_run" and (r.get("sw_run") or {}).get("status") == "pass":
            cell["csim"] = True
        if t == "rtl_sim" and (r.get("rtl_sim") or {}).get("status") == "pass":
            cell["cosim"] = True
            c = (r.get("rtl_sim") or {}).get("kernel_runtime_cycles")
            if c:
                cell["cyc_cosim"] = int(c)
    # normalize + apply degeneracy guard symmetrically (using synth-tier resources)
    out = {}
    for arm, benches in arms.items():
        cells = {}
        for b, c in benches.items():
            res = c.get("_res") or (None, None, None)
            deg = _deg(c.get("cyc_synth"), res[0], res[1], res[2])
            cells[b] = {"synth": c.get("synth", False),
                        "csim": c.get("csim", False) and not deg,
                        "cosim": c.get("cosim", False) and not deg,
                        "degenerate": deg,
                        "cyc_synth": c.get("cyc_synth"),
                        "cyc_csim": c.get("cyc_synth") if c.get("csim") else None,
                        "cyc_cosim": c.get("cyc_cosim") if not deg else None}
        out[arm] = cells
    return out


def tally(arm):
    n = len(arm)
    return (sum(v["synth"] for v in arm.values()),
            sum(v["csim"] for v in arm.values()),
            sum(v["cosim"] for v in arm.values()), n)


def main():
    ROOT = "/eagle/argonne_tpc/lyb/runs"
    ours = {
        "7B base (single)": parse_single(f"{ROOT}/eval26/7b_base"),
        "7B+GRPO (single)": parse_single(f"{ROOT}/eval26/7b_grpo_base"),
        "7B+GRPO (multistep)": parse_multistep(f"{ROOT}/eval26_multistep/7b_grpo_base"),
        "7B+step-GRPO (multistep)": parse_multistep(f"{ROOT}/eval26_multistep/7b_step_grpo"),
    }
    comm = parse_commercial("/home/lyb/schema_records.jsonl")

    print("## Per-arm tallies (polybench-26)\n")
    print(f"{'arm':34s} {'synth':>7s} {'csim':>7s} {'cosim':>7s} {'med cosim cyc':>14s}")
    allarms = list(ours.items()) + [(f"[C] {a}", d) for a, d in sorted(comm.items())]
    for name, arm in allarms:
        if not arm:
            print(f"{name:34s}   (pending)")
            continue
        s, c, co, n = tally(arm)
        cyc = sorted(v["cyc_cosim"] for v in arm.values() if v["cosim"] and v["cyc_cosim"])
        mc = f"{median(cyc):,.0f}" if cyc else "-"
        print(f"{name:34s} {s:>3d}/{n:<3d} {c:>3d}/{n:<3d} {co:>3d}/{n:<3d} {mc:>14s}")

    # --- Task 2: strict per-benchmark-intersection cosim QoR ---
    # For each pair (our best arm vs each commercial arm), median cosim cycles
    # over benchmarks BOTH pass cosim; report geomean speedup our/theirs.
    import math
    ref = ours["7B+GRPO (single)"]
    print("\n## Strict per-benchmark-intersection cosim QoR: 7B+GRPO(single) vs each commercial arm")
    print(f"{'commercial arm':38s} {'n∩':>4s} {'7B med':>10s} {'comm med':>10s} {'geomean 7B/comm':>16s}")
    for arm in sorted(comm):
        inter = [b for b in ref if ref[b]["cosim"] and comm[arm].get(b, {}).get("cosim")
                 and ref[b]["cyc_cosim"] and comm[arm][b].get("cyc_cosim")]
        if not inter:
            print(f"{arm:38s}    0        -          -                -")
            continue
        our_m = median(ref[b]["cyc_cosim"] for b in inter)
        com_m = median(comm[arm][b]["cyc_cosim"] for b in inter)
        gm = math.exp(sum(math.log(ref[b]["cyc_cosim"] / comm[arm][b]["cyc_cosim"]) for b in inter) / len(inter))
        # gm<1 => 7B fewer cycles (faster)
        print(f"{arm:38s} {len(inter):>4d} {our_m:>10,.0f} {com_m:>10,.0f} {gm:>15.2f}x")


if __name__ == "__main__":
    main()
