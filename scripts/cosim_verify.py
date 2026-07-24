"""Cosim-verify 7B-generated kernels that passed csim but were never cosim'd
(the rodinia 17-suite ships only csim testbenches; try cosim with the same tb).

CPU-only (Vitis), independent of the GPU jobs. Run on the login node with nice.

Usage:
  source env_llmhls.sh && source env_vitis.sh
  HLS_WORK_ROOT=/tmp/cosim_$USER python cosim_verify.py \
      --eval /eagle/argonne_tpc/lyb/runs/eval17/7b_grpo_base \
      --tasks /eagle/argonne_tpc/lyb/projects/llm-hls/kernels_c2hls \
      --out /eagle/argonne_tpc/lyb/runs/cosim17_7b_grpo_base.json \
      [--only kmeans,aes,knn] [--timeout 1800]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/lyb/code_translation_c2hls")
import hls_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval dir with <bench>/<bench>_generated.cpp")
    ap.add_argument("--tasks", required=True, help="task dirs with tb/header/aux (kernels_c2hls)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma list of benchmarks to limit to")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    hls_eval.COSIM_TIMEOUT = args.timeout
    evald = Path(args.eval)
    tasksd = Path(args.tasks)
    only = set(args.only.split(",")) if args.only else None

    results = {}
    for bd in sorted(p for p in evald.iterdir() if p.is_dir()):
        bench = bd.name
        if only and bench not in only:
            continue
        gen = bd / f"{bench}_generated.cpp"
        tdir = tasksd / bench
        tjson = tdir / "task.json"
        if not gen.exists() or not tjson.exists():
            continue
        res_f = bd / f"{bench}_results.json"
        prior = json.loads(res_f.read_text()) if res_f.exists() else {}
        if not (prior.get("csim") or {}).get("passed"):
            continue  # only verify designs that already pass csim
        if (prior.get("cosim") or {}).get("ran"):
            continue  # already cosim'd during eval

        task = json.loads(tjson.read_text())
        hls_code = gen.read_text()
        header_name = next((r for r in task.get("aux_files", []) if r.endswith(".h")), "kernel.h")
        header_code = (tdir / header_name).read_text() if (tdir / header_name).exists() else ""
        tb_code = (tdir / "tb.cpp").read_text()
        extra = []
        for rel in task.get("aux_files", []):
            p = tdir / rel
            if p.exists():
                extra.append({"path": rel, "content": p.read_text()})

        print(f"=== cosim: {bench} (part {task['part']} @ {task['clock_period_ns']}ns) ===", flush=True)
        t0 = time.monotonic()
        try:
            r = hls_eval.run_cosim(
                hls_code=hls_code, testbench_code=tb_code,
                header_code=header_code, header_name=header_name,
                top_function=task["top"], part=task["part"],
                clock_ns=task["clock_period_ns"],
            )
        except Exception as e:
            r = {"success": False, "passed": False, "error": f"exception: {e!r}"[:300]}
        dt = time.monotonic() - t0
        passed = bool(r.get("passed"))
        results[bench] = {"cosim_passed": passed, "wall_s": round(dt, 1),
                          "cosim_cycles": r.get("cosim_cycles") or r.get("latency_cycles"),
                          "error": (r.get("error") or "")[:200] if not passed else None}
        print(f"  -> {'PASS' if passed else 'FAIL'} ({dt:.0f}s) "
              f"{results[bench]['error'] or ''}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    npass = sum(1 for v in results.values() if v["cosim_passed"])
    print(f"\n=== COSIM SUMMARY: {npass}/{len(results)} passed -> {args.out} ===")


if __name__ == "__main__":
    main()
