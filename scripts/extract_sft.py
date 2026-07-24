"""Extract SFT training data from c2hls run results.

Walks results roots (each containing <bench>/<bench>_history.json + _results.json),
filters trajectories by success/csim, and writes a messages-format JSONL that
train_sft.py consumes directly.

Modes:
  final       one single-turn sample per benchmark: first user prompt ->
              final generated HLS code in a ```cpp fence (default; cleanest SFT v1)
  trajectory  full multi-turn chat history as one sample (repair-loop behavior)

Usage:
  python extract_sft.py --results /home/lyb/code_translation_c2hls/results \
      --out /eagle/argonne_tpc/lyb/data/hls_sft_v1.jsonl \
      --model-tag <who-generated-these> [--mode final] [--require-csim]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_bench(bench_dir: Path):
    name = bench_dir.name
    hist_f = bench_dir / f"{name}_history.json"
    res_f = bench_dir / f"{name}_results.json"
    if not hist_f.exists() or not res_f.exists():
        return None
    history = json.loads(hist_f.read_text())
    results = json.loads(res_f.read_text())
    return name, history, results


def latency_ratio(results: dict):
    comp = (results.get("comparison") or {}).get("comparison") or {}
    lat = comp.get("latency_ns") or comp.get("latency_cycles") or {}
    return lat.get("ratio")


def make_sample(name, history, results, mode, model_tag, results_root):
    hls_code = results.get("hls_code")
    if not hls_code:
        return None

    csim = (results.get("csim") or {})
    meta = {
        "benchmark": name,
        "csim_passed": bool(csim.get("passed")),
        "cosim_passed": bool((results.get("cosim") or {}).get("passed")),
        "latency_ratio_vs_gt": latency_ratio(results),
        "model": model_tag,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(results_root),
        "mode": mode,
    }

    if mode == "trajectory":
        messages = [m for m in history if m.get("role") in ("system", "user", "assistant")]
        if not any(m["role"] == "assistant" for m in messages):
            return None
        return {"messages": messages, **meta}

    # mode == "final": first user prompt -> final accepted code
    first_user = next((m for m in history if m.get("role") == "user"), None)
    system = next((m for m in history if m.get("role") == "system"), None)
    if first_user is None:
        return None
    messages = ([system] if system else []) + [
        first_user,
        {"role": "assistant", "content": f"```cpp\n{hls_code.strip()}\n```"},
    ]
    return {"messages": messages, **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="one or more results roots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["final", "trajectory"], default="final")
    ap.add_argument("--model-tag", default="unknown",
                    help="model that produced these runs (results JSONs don't store it)")
    ap.add_argument("--require-csim", action="store_true",
                    help="keep only trajectories whose final code passed csim")
    ap.add_argument("--allow-failed", action="store_true",
                    help="also keep runs with success=false (for preference/DPO data later)")
    args = ap.parse_args()

    kept, skipped = [], []
    for root in map(Path, args.results):
        for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            loaded = load_bench(bench_dir)
            if not loaded:
                continue
            name, history, results = loaded
            if not results.get("success") and not args.allow_failed:
                skipped.append((name, "run failed"))
                continue
            if args.require_csim and not (results.get("csim") or {}).get("passed"):
                skipped.append((name, "csim not passed"))
                continue
            sample = make_sample(name, history, results, args.mode, args.model_tag, root)
            if sample is None:
                skipped.append((name, "no code/prompt in history"))
                continue
            kept.append(sample)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s in kept:
            f.write(json.dumps(s) + "\n")

    print(f"wrote {len(kept)} samples -> {out}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    if kept:
        n_csim = sum(1 for s in kept if s["csim_passed"])
        print(f"csim-passed: {n_csim}/{len(kept)}")


if __name__ == "__main__":
    main()
