"""Extract csynth/csim/cosim pass counts + median latency for the GENERATED kernel
across run arms, straight from each <bench>/<bench>_results.json. Applies the
degeneracy guard (do-nothing kernels: dsp==0 and lut<threshold) so empty kernels
don't count as passes. Cosim is read from the pipeline's own inline cosim_status."""
import json, sys, statistics
from pathlib import Path

def _stat(x):
    try: return int(str(x))
    except Exception: return 0

def is_degenerate(rep):
    if not rep: return False
    dsp = _stat(rep.get("dsp")); lut = _stat(rep.get("lut"))
    return dsp == 0 and lut < 200  # do-nothing / trivial pass-through

def gen_ok(status):
    # status may be a str ("passed") or dict {"generated": "passed", ...}
    if isinstance(status, dict): return status.get("generated") == "passed"
    return status == "passed"

def parse_arm(arm_dir):
    arm = Path(arm_dir)
    rows = {}
    for bd in sorted(arm.iterdir()):
        if not bd.is_dir(): continue
        rf = list(bd.glob("*_results.json"))
        if not rf: continue
        d = json.loads(rf[0].read_text())
        rep = d.get("synth_report") or {}
        deg = is_degenerate(rep)
        csynth = bool(rep.get("latency_cycles")) and not deg
        csim = gen_ok(d.get("csim_status")) and not deg
        cosim = gen_ok(d.get("cosim_status")) and not deg
        cyc = rep.get("latency_cycles") if csynth else None
        rows[bd.name] = dict(csynth=csynth, csim=csim, cosim=cosim, cyc=cyc, deg=deg)
    return rows

def summarize(name, rows):
    n = len(rows)
    csynth = sum(r["csynth"] for r in rows.values())
    csim = sum(r["csim"] for r in rows.values())
    cosim = sum(r["cosim"] for r in rows.values())
    deg = sum(r["deg"] for r in rows.values())
    cyc = [r["cyc"] for r in rows.values() if r["csim"] and r["cyc"]]
    med = int(statistics.median(cyc)) if cyc else None
    print(f"{name:26s} n={n:3d} | csynth {csynth:2d} | csim {csim:2d} | cosim {cosim:2d} | "
          f"med-csim-cyc {str(med):>9s} | degen {deg}")
    return dict(n=n, csynth=csynth, csim=csim, cosim=cosim, med=med, deg=deg)

if __name__ == "__main__":
    for arm in sys.argv[1:]:
        summarize(Path(arm).parent.name + "/" + Path(arm).name, parse_arm(arm))
