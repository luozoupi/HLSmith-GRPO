"""Convert c2hls benchmarks into hlsenv GRPO task dirs.

For each /home/lyb/code_translation_c2hls/benchmarks/<name>/ this creates
kernels_c2hls/<name>/ with:
  kernel.cpp  reference = preferred GT variant (baseline QoR source)
  tb.cpp      the benchmark's golden testbench (csim gate)
  spec.md     the SAME translation prompt c2hls uses (plain.cpp embedded)
  task.json   top/part/clock/budgets/aux list, baseline null until baselined
  <aux>       header + support files, relative paths preserved

Then baseline each task:  python -m hlsenv baseline kernels_c2hls/<name>

Usage: python make_c2hls_tasks.py [bench ...]   (default: all with metadata.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

C2HLS = Path("/home/lyb/code_translation_c2hls")
OUT_ROOT = Path("/eagle/argonne_tpc/lyb/projects/llm-hls/kernels_c2hls")

sys.path.insert(0, str(C2HLS))
from prompt_c2hls import q_translate_c_to_hls  # noqa: E402

try:
    from c2hls import BENCHMARK_HINTS  # light dict; c2hls import needs openai present
except Exception:
    BENCHMARK_HINTS = {}

PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33
# 50% of xcu280 device resources
BUDGETS = {"lut": 651840, "ff": 1303680, "dsp": 4512, "bram": 2016}


def convert(bench_dir: Path) -> str | None:
    meta_f = bench_dir / "metadata.json"
    if not meta_f.exists():
        return None
    meta = json.loads(meta_f.read_text())
    name = meta["benchmark"]
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    gt_file = meta.get("preferred_gt_file") or meta.get("gold_hls_baseline_file") or "hls_baseline.cpp"
    (out / "kernel.cpp").write_text((bench_dir / gt_file).read_text())
    (out / "tb.cpp").write_text((bench_dir / meta.get("testbench_file", "testbench.cpp")).read_text())

    aux: list[str] = []
    header_name = meta.get("header_file")
    header_code = ""
    if header_name and (bench_dir / header_name).exists():
        header_code = (bench_dir / header_name).read_text()
        (out / header_name).write_text(header_code)
        aux.append(header_name)
    # metadata's support_files is often empty even when a support/ tree exists
    # (GT kernels include e.g. "support/common/mc.h") — sweep both.
    support_rels = list(meta.get("support_files") or [])
    support_dir = bench_dir / "support"
    if support_dir.is_dir():
        support_rels.extend(
            str(p.relative_to(bench_dir)) for p in support_dir.rglob("*") if p.is_file()
        )
    for rel in dict.fromkeys(support_rels):  # dedupe, keep order
        src = bench_dir / rel
        if src.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text())
            aux.append(rel)

    plain = (bench_dir / meta.get("plain_c_file", "plain.cpp")).read_text()
    hint = BENCHMARK_HINTS.get(name, "")
    if isinstance(hint, (list, tuple)):
        hint = "\n".join(str(h) for h in hint)
    context = (hint + "\n" if hint else "") + \
        f"The top-level function must be `{meta.get('hls_top', 'workload')}`."
    (out / "spec.md").write_text(
        q_translate_c_to_hls.format(
            benchmark_context=context, header_code=header_code, c_code=plain,
        )
    )

    task = {
        "name": name,
        "top": meta.get("hls_top", "workload"),
        "part": PART,
        "clock_period_ns": CLOCK_NS,
        "timeout_s": 600,
        "csim": bool(meta.get("supports_csim", True)),
        "baseline": None,
        "budgets": BUDGETS,
        "aux_files": aux,
        "source_benchmark": str(bench_dir),
        "gt_file": gt_file,
    }
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return name


def main():
    wanted = set(sys.argv[1:])
    made = []
    for bench_dir in sorted(p for p in (C2HLS / "benchmarks").iterdir() if p.is_dir()):
        if wanted and bench_dir.name not in wanted:
            continue
        name = convert(bench_dir)
        if name:
            made.append(name)
    print(f"generated {len(made)} tasks under {OUT_ROOT}:")
    for n in made:
        print(" ", n)


if __name__ == "__main__":
    main()
