"""Load HLS task specs from kernels/<name>/ directories."""

from __future__ import annotations

import json
from pathlib import Path


def load_task(task_dir: str | Path) -> dict:
    """A task dir holds task.json, kernel.cpp (reference), tb.cpp (golden), spec.md.

    task.json may list `aux_files` (relative paths of headers/support files in the
    task dir); their contents are loaded into task["aux_files"] and materialized
    into every synthesis scratch dir.
    """
    task_dir = Path(task_dir)
    task = json.loads((task_dir / "task.json").read_text())
    task["dir"] = str(task_dir)
    task["tb_src"] = (task_dir / "tb.cpp").read_text()
    task["ref_src"] = (task_dir / "kernel.cpp").read_text()
    task["prompt"] = (task_dir / "spec.md").read_text()
    task["aux_files"] = {
        rel: (task_dir / rel).read_text() for rel in task.get("aux_files") or []
    }
    return task


def load_tasks(kernels_dir: str | Path) -> dict[str, dict]:
    """All tasks under kernels/, keyed by name."""
    tasks = {}
    for p in sorted(Path(kernels_dir).iterdir()):
        if (p / "task.json").exists():
            t = load_task(p)
            tasks[t["name"]] = t
    if not tasks:
        raise FileNotFoundError(f"no task.json found under {kernels_dir}")
    return tasks


def save_baseline(task_dir: str | Path, qor: dict) -> None:
    """Write the reference implementation's QoR into task.json as the baseline."""
    path = Path(task_dir) / "task.json"
    task = json.loads(path.read_text())
    task["baseline"] = qor
    path.write_text(json.dumps(task, indent=2) + "\n")
