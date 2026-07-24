import json
import sys
from pathlib import Path

sys.path.insert(0, "/eagle/argonne_tpc/lyb/projects/llm-hls")
from hlsenv.qor import is_degenerate

top = ["hlsfactory_gemm", "hlsfactory_cholesky", "hlsfactory_trmm", "hlsfactory_3mm",
       "hlsfactory_fdtd-2d", "hlsfactory_jacobi-1d", "hlsfactory_atax",
       "hlsfactory_covariance", "hlsfactory_syrk", "hlsfactory_2mm"]
d = Path("/eagle/argonne_tpc/lyb/runs/eval26_skills/grpo_multi_curated")


def gi(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


print(f"{'bench':22s} {'lat':>11s} {'dsp':>6s} {'lut':>8s} {'ff':>8s}  verdict")
real = []
for b in top:
    f = d / b / f"{b}_multistep_results.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    best = None
    for s in (r.get("optimization_history") or []):
        rep = s.get("report") or {}
        lat = rep.get("latency_cycles")
        if s.get("success") and lat and (s.get("csim") or {}).get("passed"):
            lat = int(lat)
            if best is None or lat < best["lat"]:
                best = {"lat": lat, "dsp": rep.get("dsp"), "lut": rep.get("lut"), "ff": rep.get("ff")}
    if not best:
        continue
    deg = is_degenerate({"lat_worst": best["lat"], "dsp": gi(best["dsp"]),
                         "lut": gi(best["lut"]), "ff": gi(best["ff"])})
    verdict = "DEGENERATE (skip)" if deg else "real -> cosim"
    print(f"{b:22s} {best['lat']:>11,} {str(best['dsp']):>6s} {str(best['lut']):>8s} {str(best['ff']):>8s}  {verdict}")
    if not deg:
        real.append(b)
print("\ncosim targets:", ",".join(real))
