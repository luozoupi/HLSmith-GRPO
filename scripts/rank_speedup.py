import json
import sys
from pathlib import Path

sys.path.insert(0, "/eagle/argonne_tpc/lyb/projects/llm-hls/scripts")
from compare_arms import parse_commercial, _norm

comm = parse_commercial("/home/lyb/schema_records.jsonl")
best_comm = {}
for arm, bs in comm.items():
    for b, c in bs.items():
        if c["cosim"] and c.get("cyc_cosim") and (b not in best_comm or c["cyc_cosim"] < best_comm[b][1]):
            best_comm[b] = (arm.replace("enh__", ""), c["cyc_cosim"])

rows = []
d = Path("/eagle/argonne_tpc/lyb/runs/eval26_skills/grpo_multi_curated")
for bd in d.iterdir():
    f = bd / f"{bd.name}_multistep_results.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    best = None
    for s in (r.get("optimization_history") or []):
        rep = s.get("report") or {}
        lat = rep.get("latency_cycles")
        if s.get("success") and lat and (s.get("csim") or {}).get("passed"):
            lat = int(lat)
            if best is None or lat < best:
                best = lat
    gtlat = (r.get("baseline_report") or {}).get("latency_cycles")
    if best and gtlat:
        b = _norm(bd.name)
        sp = gtlat / best
        bc = best_comm.get(b)
        vc = best / bc[1] if bc else None
        rows.append((bd.name, best, int(gtlat), sp, bc[0] if bc else "-", bc[1] if bc else None, vc))

rows.sort(key=lambda x: -x[3])
print(f"{'bench':20s} {'7B cyc':>11s} {'gold cyc':>11s} {'vs gold':>8s} {'vs best commercial':>28s}")
for name, best, gt, sp, ca, cc, vc in rows:
    vcs = f"{ca[:18]} {vc:.2f}x" if vc else "-"
    star = "  <<" if (sp >= 2 or (vc and vc <= 1.0)) else ""
    print(f"{name:20s} {best:>11,} {gt:>11,} {sp:>7.2f}x {vcs:>28s}{star}")
print("\n<< = superb (>=2x vs gold, or beats best commercial) -> cosim-verify these")
