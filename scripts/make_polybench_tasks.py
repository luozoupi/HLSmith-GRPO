"""Convert polybench (benchmarks_hlsfactory) benchmarks into hlsenv GRPO task dirs
for GRPO-v2 (same domain as the SFT corpus). Mirrors make_c2hls_tasks.py but points
at benchmarks_hlsfactory and adapts its metadata key names.

For each data/benchmarks_hlsfactory/<name>/ this writes kernels_polybench/<name>/:
  kernel.cpp  reference = hls_baseline.cpp (baseline QoR source for the reward)
  tb.cpp      the benchmark's csim testbench (correctness gate)
  spec.md     the SAME translation prompt c2hls uses (plain.cpp embedded)
  task.json   top/part/clock/budgets/aux, baseline null until baselined
  <header>    header file (aux)

Then baseline each:  python -m hlsenv baseline kernels_polybench/<name>

Usage: python make_polybench_tasks.py <bench_dir_name> [<bench_dir_name> ...]
       (bench_dir_name = the benchmarks_hlsfactory dir, e.g. hlsfactory_jacobi-2d)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

C2HLS = Path("/home/lyb/code_translation_c2hls")
BENCH_ROOT = Path("/eagle/argonne_tpc/lyb/data/benchmarks_hlsfactory")
OUT_ROOT = Path("/eagle/argonne_tpc/lyb/projects/llm-hls/kernels_polybench")

sys.path.insert(0, str(C2HLS))
from prompt_c2hls import q_translate_c_to_hls  # noqa: E402
# Use the pipeline's EXACT context builder so GRPO-training spec.md matches the
# eval prompt distribution — this is what emits "Required HLS wrapper top
# function: `kernel_2mm`", overriding the template's workload() default so the
# generated kernel links against the kernel_2mm testbench (reward gate).
from c2hls import _build_benchmark_context  # noqa: E402

PART = "xcu280-fsvh2892-2L-e"
CLOCK_NS = 3.33
BUDGETS = {"lut": 651840, "ff": 1303680, "dsp": 4512, "bram": 2016}  # 50% of xcu280


def convert(bench_dir: Path) -> str | None:
    meta_f = bench_dir / "metadata.json"
    if not meta_f.exists():
        return None
    meta = json.loads(meta_f.read_text())
    name = meta["benchmark"]
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    # baseline reference the reward measures speedup against
    gt_file = meta.get("gold_hls_baseline_file") or "hls_baseline.cpp"
    (out / "kernel.cpp").write_text((bench_dir / gt_file).read_text())
    (out / "tb.cpp").write_text((bench_dir / meta.get("testbench_file", "testbench.cpp")).read_text())

    aux: list[str] = []
    header_name = meta.get("header_file")
    header_code = ""
    if header_name and (bench_dir / header_name).exists():
        header_code = (bench_dir / header_name).read_text()
        (out / header_name).write_text(header_code)
        aux.append(header_name)
    for rel in dict.fromkeys(meta.get("support_files") or []):
        src = bench_dir / rel
        if src.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text())
            aux.append(rel)

    plain = (bench_dir / meta.get("plain_c_file", "plain.cpp")).read_text()
    top = meta.get("translated_hls_top") or meta.get("hls_top") or meta.get("kernel_top") or "workload"
    gold_name = meta.get("gold_hls_source_file")
    gold_code = (bench_dir / gold_name).read_text() if gold_name and (bench_dir / gold_name).exists() else ""
    # identical bulleted context the eval pipeline builds (incl. "Required HLS
    # wrapper top function: `<top>`") so training and eval prompts match exactly.
    context = _build_benchmark_context(meta, header_name, header_code, plain, gold_code)
    (out / "spec.md").write_text(
        q_translate_c_to_hls.format(benchmark_context=context, header_code=header_code, c_code=plain)
    )

    task = {
        "name": name, "top": top, "part": PART, "clock_period_ns": CLOCK_NS,
        "timeout_s": 600, "csim": bool(meta.get("supports_csim", True)),
        "baseline": None, "budgets": BUDGETS, "aux_files": aux,
        "source_benchmark": str(bench_dir), "gt_file": gt_file,
    }
    (out / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return name


def main():
    wanted = sys.argv[1:]
    dirs = [BENCH_ROOT / w for w in wanted] if wanted else \
           [p for p in sorted(BENCH_ROOT.iterdir()) if (p / "metadata.json").exists()]
    made = [convert(d) for d in dirs if convert(d)]
    print(f"generated {len(made)} tasks under {OUT_ROOT}:")
    for n in made:
        print(" ", n)


if __name__ == "__main__":
    main()
