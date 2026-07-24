"""LLM-as-judge calibration against REAL SDK labels (strongest SDK-free proxy).

Reuses the DPO matched pairs (chosen = SDK-pass, rejected = SDK-fail) so the judge's
predictive power is directly comparable to the static-proxy calibration
(eval_methods_calibration.py). For each pair we show the judge the CUDA source + both
CSL candidates (order randomized to control position bias) and ask which is more likely
to COMPILE + PASS simulation. Accuracy = fraction where the judge picks the true pass.

A random judge scores ~0.50; the static proxies' pairwise "prefers chosen" was ~0.04-0.12.

Usage (on the node serving the judge; needs `openai`):
  python llm_judge_calibration.py --dpo <dpo.jsonl> --base-url http://127.0.0.1:8000/v1 \
      --model <served> --n 200 --out <out.json>
"""
from __future__ import annotations
import argparse, json, re, random
from collections import defaultdict

def fenced(text, lang):
    m = re.search(rf"```{lang}\s*\n(.*?)```", text or "", re.DOTALL)
    return m.group(1).strip() if m else None

def extract_csl(t):
    return fenced(t, "csl") or (t or "").strip()

def cuda_from_prompt(msgs):
    joined = "\n".join(m.get("content", "") if isinstance(m.get("content"), str) else ""
                       for m in msgs)
    return fenced(joined, "cuda") or ""

SYS = ("You are an expert in CUDA GPU programming and Cerebras CSL. You assess which of two "
       "CSL translations of a CUDA kernel is more likely to COMPILE with cslc and PASS functional "
       "RTL/simulator verification. Judge correctness and launch-contract compliance, not style.")

TMPL = """A CUDA kernel was translated into a Cerebras CSL compute file. Below are two candidate
translations of the SAME kernel. Exactly one is known to compile and pass simulation; the other
fails. Decide which is more likely to be the passing one.

## CUDA source
```cuda
{cuda}
```

## Candidate A
```csl
{a}
```

## Candidate B
```csl
{b}
```

Respond with EXACTLY one character on the first line: A or B. Then optionally a one-line reason."""

def parse_pick(reply):
    if not reply: return None
    m = re.search(r"\b([AB])\b", reply.strip()[:20])
    return m.group(1) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-csl-chars", type=int, default=6000)
    ap.add_argument("--max-cuda-chars", type=int, default=4000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dpo)]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    n = correct = wrong = unparsed = 0
    picked_A = 0
    per_kernel = defaultdict(lambda: [0, 0])  # kernel -> [correct, total]
    used = 0
    for r in rows:
        if used >= args.n: break
        cuda = cuda_from_prompt(r.get("prompt", []))[: args.max_cuda_chars]
        ch = extract_csl(r.get("chosen", ""))[: args.max_csl_chars]
        rj = extract_csl(r.get("rejected", ""))[: args.max_csl_chars]
        if not ch or not rj:
            continue
        used += 1
        kern = (r.get("metadata") or {}).get("kernel", "?")
        chosen_is_A = rng.random() < 0.5          # randomize position
        a, b = (ch, rj) if chosen_is_A else (rj, ch)
        prompt = TMPL.format(cuda=cuda or "(CUDA source unavailable)", a=a, b=b)
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt}],
                max_tokens=64, temperature=0.0, timeout=300)
            pick = parse_pick(resp.choices[0].message.content or "")
        except Exception as e:
            pick = None
            print(f"[err] {kern}: {type(e).__name__}: {e}")
        n += 1
        if pick is None:
            unparsed += 1
            continue
        if pick == "A":
            picked_A += 1
        judged_chosen = (pick == "A" and chosen_is_A) or (pick == "B" and not chosen_is_A)
        per_kernel[kern][1] += 1
        if judged_chosen:
            correct += 1; per_kernel[kern][0] += 1
        else:
            wrong += 1
        if n % 25 == 0:
            print(f"  {n} judged | acc={correct/max(1,correct+wrong):.3f} | posA={picked_A/n:.2f}")

    decided = correct + wrong
    out = {
        "judge_model": args.model, "n_attempted": n, "n_decided": decided,
        "unparsed": unparsed,
        "accuracy": round(correct / decided, 3) if decided else None,
        "net_signal": round((correct - wrong) / decided, 3) if decided else None,
        "position_bias_pickedA": round(picked_A / n, 3) if n else None,
        "per_kernel_accuracy": {k: round(c / t, 3) for k, (c, t) in sorted(per_kernel.items()) if t},
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
