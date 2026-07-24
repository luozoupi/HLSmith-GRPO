"""Aggregate schema_records.jsonl (Opus, mode x skills arms) into a baseline table
comparable with our RL'ed-model sweeps: per arm — synth/csim/cosim rates and
median worst-case latency on passing benchmarks.

Usage: python opus_baseline_table.py /home/lyb/schema_records.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median


def main(path: str):
    per = defaultdict(lambda: defaultdict(dict))  # arm -> bench -> facts
    for line in open(path):
        r = json.loads(line)
        arm = (r.get("implementation") or {}).get("origin_version") or "unlabeled"
        bench = "/".join((r.get("problem") or {}).get("group_path") or ["?"])
        cell = per[arm][bench]
        t = r.get("report_type")
        if t == "hls_synth" and "hls_synth" in r:
            hs = r["hls_synth"]
            ok = hs.get("status") == "pass"
            cell["synth"] = cell.get("synth", False) or ok
            if ok:
                lat = (hs.get("PerformanceEstimates", {})
                         .get("SummaryOfOverallLatency", {})
                         .get("Worst-caseLatency"))
                try:
                    lat = int(float(lat))
                    if lat and lat < cell.get("lat", float("inf")):
                        cell["lat"] = lat
                except (TypeError, ValueError):
                    pass
        elif t == "sw_run":
            cell["csim"] = cell.get("csim", False) or (r.get("sw_run", {}).get("status") == "pass")
        elif t == "rtl_sim":
            cell["cosim"] = cell.get("cosim", False) or (r.get("rtl_sim", {}).get("status") == "pass")

    print(f"{'arm':40s} {'n':>3} {'synth':>6} {'csim':>6} {'cosim':>6} {'med_lat_cyc':>12}")
    for arm in sorted(per):
        cells = per[arm]
        n = len(cells)
        s = sum(1 for c in cells.values() if c.get("synth"))
        cs = sum(1 for c in cells.values() if c.get("csim"))
        co = sum(1 for c in cells.values() if c.get("cosim"))
        lats = sorted(c["lat"] for c in cells.values() if "lat" in c)
        med = f"{median(lats):,.0f}" if lats else "-"
        print(f"{arm:40s} {n:>3} {s:>6} {cs:>6} {co:>6} {med:>12}")


if __name__ == "__main__":
    main(sys.argv[1])
