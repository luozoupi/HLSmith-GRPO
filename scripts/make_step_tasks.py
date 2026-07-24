"""Generate step-role GRPO tasks: (validated design, optimization-step instruction).

For every baselined task in kernels_c2hls x each optimization step, emit
kernels_c2hls_steps/<bench>__<step>/ where:
  kernel.cpp   the input design (= source task's reference; baseline QoR applies)
  spec.md      the c2hls step prompt (tiling/pipeline/...) with the design embedded
  task.json    copied, renamed; baseline RETAINED so hlsenv's shaped reward
               scores csim-gated speedup OVER THE INPUT DESIGN
  tb.cpp/aux   copied verbatim (same golden testbench gate)

Usage: python make_step_tasks.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/home/lyb/code_translation_c2hls")
from prompt_c2hls import OPTIMIZATION_PROMPTS  # noqa: E402

SRC = Path("/eagle/argonne_tpc/lyb/projects/llm-hls/kernels_c2hls")
DST = Path("/eagle/argonne_tpc/lyb/projects/llm-hls/kernels_c2hls_steps")


def fmt_report(baseline: dict) -> str:
    keys = ("lat_worst", "ii_max", "lut", "ff", "dsp", "bram", "clk_est_ns")
    return "\n".join(f"  {k}: {baseline.get(k)}" for k in keys if baseline.get(k) is not None)


def main():
    made = 0
    for src_dir in sorted(p for p in SRC.iterdir() if (p / "task.json").exists()):
        task = json.loads((src_dir / "task.json").read_text())
        if not task.get("baseline"):
            continue
        if task.get("timeout_s", 600) > 900:
            continue
        header_code = ""
        for rel in task.get("aux_files") or []:
            if rel.endswith(".h"):
                header_code = (src_dir / rel).read_text()
                break
        current_code = (src_dir / "kernel.cpp").read_text()
        for step, template in OPTIMIZATION_PROMPTS.items():
            name = f"{task['name']}__{step}"
            out = DST / name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dir / "kernel.cpp", out / "kernel.cpp")
            shutil.copy(src_dir / "tb.cpp", out / "tb.cpp")
            for rel in task.get("aux_files") or []:
                dst_f = out / rel
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_dir / rel, dst_f)
            (out / "spec.md").write_text(template.format(
                synth_report=fmt_report(task["baseline"]),
                header_code=header_code,
                current_code=current_code,
            ))
            t = dict(task)
            t["name"] = name
            t["source_task"] = task["name"]
            t["step"] = step
            (out / "task.json").write_text(json.dumps(t, indent=2) + "\n")
            made += 1
    print(f"generated {made} step tasks under {DST}")


if __name__ == "__main__":
    main()
