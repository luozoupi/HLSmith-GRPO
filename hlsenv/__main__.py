"""CLI: python -m hlsenv {baseline|synth} <task_dir> [--kernel FILE]

  baseline  synthesize the reference kernel.cpp and write its QoR into task.json
  synth     synthesize a kernel against a task and print status/QoR/reward
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .reward import shaped_reward
from .runner import STATUS_OK, run_synthesis
from .tasks import load_task, save_baseline


def _synth_for_task(task: dict, kernel_src: str, keep_scratch: bool = False):
    return run_synthesis(
        kernel_src,
        task["tb_src"],
        task["top"],
        part=task["part"],
        clock_period_ns=task["clock_period_ns"],
        timeout_s=task.get("timeout_s", 300),
        csim=task.get("csim", True),
        keep_scratch=keep_scratch,
        aux_files=task.get("aux_files") or None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(prog="hlsenv")
    ap.add_argument("command", choices=["baseline", "synth"])
    ap.add_argument("task_dir")
    ap.add_argument("--kernel", help="kernel source to synthesize (default: task's kernel.cpp)")
    ap.add_argument("--keep-scratch", action="store_true")
    args = ap.parse_args()

    task = load_task(args.task_dir)
    kernel_src = Path(args.kernel).read_text() if args.kernel else task["ref_src"]

    result = _synth_for_task(task, kernel_src, keep_scratch=args.keep_scratch)
    print(f"status: {result.status}  wall: {result.wall_s:.1f}s")
    if result.status != STATUS_OK:
        print(result.log_tail[-2000:])
        return 1
    print(json.dumps(result.qor, indent=2))

    if args.command == "baseline":
        save_baseline(args.task_dir, result.qor)
        print(f"baseline written to {args.task_dir}/task.json")
    elif task.get("baseline"):
        print(f"reward vs baseline: {shaped_reward(result.qor, task['baseline'], task.get('budgets', {})):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
