"""CPU-only cosim verification of generated polybench kernels, using the enhanced
pipeline's EXACT cosim assembly (mirrors cosim_smoke_test.py::_load_bench_assets):
 - cosim testbench = cosim_testbench_file (separate compile unit, NOT concatenated)
 - extra_files: gold_kernel_for_cosim.cpp (compiled) + gold_hls_source.cpp (compile:False,
   materialized only — the shim #includes it; a second compile unit would duplicate-define)
 - size_overrides = cosim_size_overrides (cosim runs at shrunk sizes; full size is infeasible)

Runs on the login-node CPU (no GPU / no allocation). Use --gold-vs-gold to validate
the harness (must PASS by construction) before trusting generated-kernel verdicts.

Usage:
  source env_llmhls.sh   # then export C2HLS_VITIS_SETTINGS, C2HLS_COSIM_LIBRARY_PATH=/usr/lib64:/lib64
  python cosim_verify_enh.py --eval <dir> --benchmarks <root> --out out.json \
      [--only a,b,c] [--gold-vs-gold] [--timeout 1200]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/eagle/argonne_tpc/lyb/code_translation_c2hls_enh")
import hls_eval  # noqa: E402


def apply_overrides(code: str, overrides: dict) -> str:
    """Shrink problem-size macros for cosim (this branch's run_cosim predates
    -D size_overrides). Rewrites `#define <KEY> <n>` to the override value."""
    for key, val in (overrides or {}).items():
        code = re.sub(rf'(#define\s+{re.escape(key)}\s+)\S+', rf'\g<1>{val}', code)
    return code


def load_assets(bench_dir: Path) -> dict:
    meta = json.loads((bench_dir / "metadata.json").read_text())
    header = (bench_dir / meta["header_file"]).read_text() if meta.get("header_file") else ""
    gold_src_name = meta.get("gold_hls_source_file")
    gold_code = (bench_dir / gold_src_name).read_text() if gold_src_name else ""
    cosim_tb = meta.get("cosim_testbench_file")
    cosim_tb_code = (bench_dir / cosim_tb).read_text() if cosim_tb and (bench_dir / cosim_tb).exists() else None

    extras = []
    for name in meta.get("cosim_support_files") or []:
        p = bench_dir / name
        if p.exists():
            extras.append({"path": name, "content": p.read_text()})  # compiled unit (the shim)
    # gold source: materialize only (shim #includes it) — NOT a compile unit
    if extras and gold_src_name and (bench_dir / gold_src_name).exists():
        extras.append({"path": gold_src_name, "content": gold_code, "compile": False})

    return {
        "meta": meta, "top": meta.get("kernel_top") or meta.get("hls_top"),
        "header_name": meta.get("header_file") or "kernel.h", "header_code": header,
        "gold_code": gold_code, "cosim_tb_code": cosim_tb_code, "extras": extras,
        "overrides": meta.get("cosim_size_overrides") or {},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", help="dir with <bench>/<bench>_final.cpp (generated kernels)")
    ap.add_argument("--benchmarks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--gold-vs-gold", action="store_true", help="verify harness: DUT=gold (must pass)")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()
    hls_eval.COSIM_TIMEOUT = args.timeout

    bmroot = Path(args.benchmarks)
    only = set(args.only.split(",")) if args.only else None
    results = {}

    names = sorted(only) if only else [p.name for p in bmroot.iterdir() if (p / "metadata.json").exists()]
    for bench in names:
        bd = bmroot / bench
        if not (bd / "metadata.json").exists():
            continue
        A = load_assets(bd)
        if not A["cosim_tb_code"]:
            results[bench] = {"cosim_passed": None, "reason": "no cosim testbench"}
            continue

        if args.gold_vs_gold:
            dut = A["gold_code"]
        else:
            gen_f = Path(args.eval) / bench / f"{bench}_final.cpp"
            if not gen_f.exists():
                gen_f = Path(args.eval) / bench / f"{bench}_generated.cpp"
            if not gen_f.exists():
                results[bench] = {"cosim_passed": None, "reason": "no generated kernel"}
                continue
            dut = gen_f.read_text()

        print(f"=== cosim {bench} (top={A['top']}, overrides={A['overrides']}) ===", flush=True)
        # shrink dims for feasible RTL sim (applied to header — it owns the macros)
        header_code = apply_overrides(A["header_code"], A["overrides"])
        t0 = time.monotonic()
        try:
            r = hls_eval.run_cosim(
                dut, A["cosim_tb_code"], header_code, header_name=A["header_name"],
                top_function=A["top"], extra_files=A["extras"],
            )
        except Exception as e:
            r = {"passed": False, "error": f"exception: {e!r}"[:300]}
        dt = time.monotonic() - t0
        ok = bool(r.get("passed"))
        results[bench] = {"cosim_passed": ok, "wall_s": round(dt, 1),
                          "cosim_cycles": r.get("cosim_cycles") or r.get("latency_cycles"),
                          "error": None if ok else (r.get("error") or "")[:200]}
        print(f"  -> {'PASS' if ok else 'FAIL'} ({dt:.0f}s)  {results[bench].get('error') or ''}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    npass = sum(1 for v in results.values() if v.get("cosim_passed"))
    print(f"\n=== COSIM: {npass}/{sum(1 for v in results.values() if v.get('cosim_passed') is not None)} passed → {args.out} ===")


if __name__ == "__main__":
    main()
