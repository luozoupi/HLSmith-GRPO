"""Repair c2hls-derived tasks whose preferred GT variant fails csim/compile here.

Mirrors c2hls's own fallback: walk candidates from most- to least-optimized
(preferred_gt_file, then variants newest->oldest, then hls_baseline.cpp) and
adopt the first one that synthesizes AND passes csim in the hlsenv harness.
The adopted variant becomes the task's kernel.cpp + baseline QoR. Timeout is
raised per attempt and, if the winner needed more than the task budget, the
task's timeout_s is bumped to 2x its wall time.

Usage: python fix_c2hls_baselines.py <task_name> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hlsenv.runner import STATUS_OK, run_synthesis
from hlsenv.tasks import load_task, save_baseline

C2HLS_BENCH = Path("/home/lyb/code_translation_c2hls/benchmarks")
TASK_ROOT = Path("/eagle/argonne_tpc/lyb/projects/llm-hls/kernels_c2hls")
ATTEMPT_TIMEOUT = 1800


def candidates_for(meta: dict) -> list[str]:
    cands: list[str] = []
    if meta.get("preferred_gt_file"):
        cands.append(meta["preferred_gt_file"])
    for v in reversed(meta.get("variants") or []):
        f = v.get("file") if isinstance(v, dict) else None
        if f:
            cands.append(f)
    if meta.get("gold_hls_baseline_file"):
        cands.append(meta["gold_hls_baseline_file"])
    cands.append("hls_baseline.cpp")
    return list(dict.fromkeys(cands))


def fix(name: str) -> bool:
    task_dir = TASK_ROOT / name
    task = load_task(task_dir)
    meta = json.loads((C2HLS_BENCH / name / "metadata.json").read_text())

    for cand in candidates_for(meta):
        src_path = C2HLS_BENCH / name / cand
        if not src_path.exists():
            continue
        print(f"[{name}] trying {cand} ...", flush=True)
        result = run_synthesis(
            src_path.read_text(),
            task["tb_src"],
            task["top"],
            part=task["part"],
            clock_period_ns=task["clock_period_ns"],
            timeout_s=ATTEMPT_TIMEOUT,
            csim=task.get("csim", True),
            aux_files=task.get("aux_files") or None,
        )
        print(f"[{name}]   {cand}: {result.status} ({result.wall_s:.0f}s)", flush=True)
        if result.status == STATUS_OK:
            (task_dir / "kernel.cpp").write_text(src_path.read_text())
            save_baseline(task_dir, result.qor)
            tj = json.loads((task_dir / "task.json").read_text())
            tj["gt_file"] = cand
            if result.wall_s * 2 > tj.get("timeout_s", 600):
                tj["timeout_s"] = int(result.wall_s * 2)
            (task_dir / "task.json").write_text(json.dumps(tj, indent=2) + "\n")
            print(f"[{name}] FIXED with {cand}; timeout_s={tj['timeout_s']}", flush=True)
            return True
        if result.status == "TIMEOUT":
            print(f"[{name}]   (log tail) {result.log_tail[-300:]}", flush=True)
    print(f"[{name}] NO candidate passed — leaving baseline empty", flush=True)
    return False


if __name__ == "__main__":
    names = sys.argv[1:]
    outcome = {n: fix(n) for n in names}
    print("\n=== FIX SUMMARY ===")
    for n, ok in outcome.items():
        print(f"  {n}: {'fixed' if ok else 'UNFIXED'}")
    sys.exit(0 if all(outcome.values()) else 1)
