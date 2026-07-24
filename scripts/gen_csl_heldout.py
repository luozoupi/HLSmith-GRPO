"""Generate CSL for held-out kernels against a local vLLM OpenAI server (no Cerebras SDK).

Controlled comparison: replays the EXACT prompts the commercial models saw (stored in
`test_prompts.jsonl` by prep_csl_corpus.py), so a generated completion is directly
comparable to the commercial artifact for the same prompt. Never touches cslc/cs_python.

Prompt shape per record = [system(469-char instruction), first "Translate ..." user turn].
The served model replies with prose + a ```csl fence; we store the raw reply and the
extracted CSL. Run once per model (fine-tuned adapter, then base) to build the scoreboard.

Usage (on the node running vLLM; needs only `openai`):
  python gen_csl_heldout.py \
    --test-prompts /eagle/argonne_tpc/lyb/data/csl_sft/test_prompts.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model csl-sft \
    --out /eagle/argonne_tpc/lyb/runs/eval/gen_csl-sft.jsonl \
    --max-per-kernel 8 --max-tokens 8192
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_FENCE = re.compile(r"```(?:csl)?\s*\n(.*?)```", re.DOTALL)


def extract_csl(reply: str) -> str:
    """First ```csl fence; fall back to first generic fence; else the raw text."""
    m = re.search(r"```csl\s*\n(.*?)```", reply, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = _FENCE.search(reply)
    if m:
        return m.group(1).strip()
    return reply.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-prompts", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="served-model-name on the vLLM server")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-kernel", type=int, default=8,
                    help="cap generations per kernel to bound cost/time")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    rows = [json.loads(l) for l in open(args.test_prompts)]
    per_kernel = Counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_err = 0
    with open(args.out, "w") as g:
        for i, r in enumerate(rows):
            kern = r["kernel"]
            if per_kernel[kern] >= args.max_per_kernel:
                continue
            per_kernel[kern] += 1
            try:
                resp = client.chat.completions.create(
                    model=args.model, messages=r["prompt"],
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    timeout=args.timeout)
                reply = resp.choices[0].message.content or ""
                gen_csl = extract_csl(reply)
                has_fence = "```csl" in reply or "```" in reply
                n_ok += 1
                status = "ok"
            except Exception as e:  # keep going; record the failure
                reply, gen_csl, has_fence, status = "", "", False, f"error:{type(e).__name__}:{e}"
                n_err += 1

            g.write(json.dumps({
                "kernel": kern,
                "src_index": i,
                "served_model": args.model,
                "status": status,
                "gen_raw": reply,
                "gen_csl": gen_csl,
                "gen_has_fence": has_fence,
                "reference_csl": r["reference_csl"],       # commercial-model artifact
                "commercial_model": r.get("model"),
                "ref_cycles": r.get("cycles_send"),
                "run_dir": r.get("run_dir"),
            }) + "\n")
            g.flush()
            print(f"[{i}] {kern:24s} {status:12s} gen_csl_chars={len(gen_csl)}")

    print(f"\nDONE: {n_ok} ok, {n_err} errors -> {args.out}")
    print("per-kernel generated:", dict(per_kernel))


if __name__ == "__main__":
    main()
