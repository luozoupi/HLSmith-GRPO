"""C1 gate: run the c2hls gold reference kernels through the PATCHED hls_eval
(Vitis 2023.2 classic + U280 @ 3.33 ns) — synthesis + csim, no LLM involved.

Usage:
  source env_llmhls.sh && source env_vitis.sh
  HLS_WORK_ROOT=/tmp/c2hls_gold python validate_c2hls_gold.py gemm_ncubed aes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

C2HLS = Path("/home/lyb/code_translation_c2hls")
sys.path.insert(0, str(C2HLS))

from hls_eval import DEFAULT_CLOCK_NS, DEFAULT_PART, run_csim, run_hls_synthesis  # noqa: E402

import os

BENCH_ROOT = Path(os.getenv("C2HLS_BENCH_ROOT", str(C2HLS / "benchmarks")))


def validate(bench: str) -> bool:
    d = BENCH_ROOT / bench
    meta = json.loads((d / "metadata.json").read_text())
    gt_file = meta.get("preferred_gt_file") or meta.get("gold_hls_baseline_file") or "hls_baseline.cpp"
    hls_code = (d / gt_file).read_text()
    header_name = meta.get("header_file") or ""
    header_code = (d / header_name).read_text() if header_name and (d / header_name).exists() else ""
    tb = (d / meta.get("testbench_file", "testbench.cpp"))
    top = meta.get("hls_top") or meta.get("kernel_top") or "workload"
    extra = []
    for rel in meta.get("support_files") or []:
        p = d / rel
        if p.exists():
            extra.append({"path": rel, "content": p.read_text()})

    print(f"\n=== {bench}: gold={gt_file} top={top} part={DEFAULT_PART} clk={DEFAULT_CLOCK_NS}ns "
          f"support={len(extra)} ===", flush=True)

    synth = run_hls_synthesis(hls_code=hls_code, header_code=header_code,
                              header_name=header_name or "kernel.h",
                              top_function=top, extra_files=extra or None)
    ok = synth.get("success", False)
    print(f"  synth: {'PASS' if ok else 'FAIL'}", flush=True)
    if ok:
        r = synth["report"]
        print(f"  QoR: lat={r.get('latency_cycles')}cyc/{r.get('latency_ns')}ns "
              f"lut={r.get('lut')} ff={r.get('ff')} dsp={r.get('dsp')} bram={r.get('bram')} "
              f"fmax={r.get('fmax_mhz')}MHz slack={r.get('slack_ns')}ns")
    else:
        print(f"  error: {str(synth.get('error'))[:600]}")
        return False

    if meta.get("supports_csim") and tb.exists():
        csim = run_csim(hls_code=hls_code, testbench_code=tb.read_text(),
                        header_code=header_code, header_name=header_name or "kernel.h",
                        top_function=top, extra_files=extra or None)
        print(f"  csim: {'PASS' if csim.get('passed') else 'FAIL'} "
              f"({str(csim.get('error'))[:200] if not csim.get('passed') else 'ok'})", flush=True)
        ok = ok and csim.get("passed", False)
    return ok


if __name__ == "__main__":
    benches = sys.argv[1:] or ["gemm_ncubed", "aes"]
    results = {b: validate(b) for b in benches}
    print("\n=== C1 SUMMARY ===")
    for b, ok in results.items():
        print(f"  {b}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
