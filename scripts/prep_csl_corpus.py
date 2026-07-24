"""Prepare the CUDA->CSL historical corpus for offline SFT.

Turns the huge multi-turn agent trajectories in `sft_corpus.jsonl` into compact,
single-shot PROMPT-COMPLETION examples that fit GPU memory and train train~=serve:

  prompt     = [ system(469-char instruction), first "Translate ..." user turn ]
  completion = [ final assistant turn (the ```csl block, with its prose) ]

Why prompt-completion (not messages): TRL 1.8 SFTTrainer auto-enables
`completion_only_loss` for prompt-completion datasets (sft_trainer.py:1190), so the
loss is computed ONLY on the ~2.5K-token CSL completion, not the ~27K-token
reference-bundle prompt. train_sft.py needs no change.

Filters:
  - keep only PASSING trajectories (metadata.status == "pass" or reward == 1.0)
  - require the final assistant turn to contain a ```csl fence
  - require an identifiable first "Translate the following CUDA kernel" user turn
  - HOLD OUT whole kernels (leave-some-kernels-out) for eval vs commercial artifacts
  - LENGTH FILTER (drop, never truncate) chat-templated prompt+completion > max_seq_len
  - dedup identical (prompt, completion)

Outputs (in --out-dir):
  train.jsonl / val.jsonl          {"prompt":[...], "completion":[...]}
  train.meta.jsonl / val.meta.jsonl aligned sidecars {kernel,model,cycles_send,run_dir}
  test_prompts.jsonl               held-out records: {kernel,prompt,reference_csl,model,cycles_send,run_dir}
  prep_summary.json                all counts + token stats

Usage (login node, tokenizer only, no GPU):
  python prep_csl_corpus.py \
    --sft-jsonl /eagle/argonne_tpc/lyb/data/csl_sft/raw/corpus/sft_corpus.jsonl \
    --out-dir   /eagle/argonne_tpc/lyb/data/csl_sft \
    --model     Qwen/Qwen3-32B --max-seq-len 32768
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

TRANSLATE_MARK = "Translate the following CUDA kernel"
CSL_FENCE = "```csl"
DEFAULT_HOLDOUT = ["GEMM", "Cholesky", "Jacobi-2D-5pt", "SpMV-CSR",
                   "Tensor-Transpose-021", "Residual"]


def content_of(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


def reconstruct(rec: dict):
    """Return (system_msg, translate_user_msg, final_assistant_msg) or None."""
    msgs = rec.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        return None
    sys_msg = {"role": "system", "content": content_of(msgs[0])}
    last_asst = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    if not last_asst or CSL_FENCE not in content_of(last_asst):
        return None
    first_tr = next((m for m in msgs
                     if m.get("role") == "user" and TRANSLATE_MARK in content_of(m)), None)
    if first_tr is None:
        return None
    return (sys_msg,
            {"role": "user", "content": content_of(first_tr)},
            {"role": "assistant", "content": content_of(last_asst)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B",
                    help="tokenizer used for the length filter (train the same base)")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--holdout-kernels", default=",".join(DEFAULT_HOLDOUT))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-length-filter", action="store_true",
                    help="skip tokenizer load / length filter (fast dry run)")
    args = ap.parse_args()

    holdout = {k.strip() for k in args.holdout_kernels.split(",") if k.strip()}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = None
    if not args.no_length_filter:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)

    def n_tokens(p_msgs, c_msgs) -> int:
        # tokens of the full chat-templated conversation (== training length)
        text = tok.apply_chat_template(p_msgs + c_msgs, tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    stats = {
        "sft_jsonl": args.sft_jsonl, "model": args.model,
        "max_seq_len": args.max_seq_len, "holdout_kernels": sorted(holdout),
        "total": 0, "passing": 0, "reconstructable": 0,
        "heldout_records": 0, "train_pool": 0,
        "dropped_over_len": 0, "dropped_dup": 0,
        "kept": 0, "train": 0, "val": 0,
        "per_kernel_kept": {}, "per_kernel_heldout": {}, "token_len": {},
    }

    kept = []           # (prompt_msgs, completion_msgs, meta)
    heldout_rows = []
    seen = set()
    tok_lens = []

    with open(args.sft_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            md = rec.get("metadata", {}) or {}
            if not (md.get("status") == "pass" or md.get("reward") == 1.0):
                continue
            stats["passing"] += 1
            r = reconstruct(rec)
            if r is None:
                continue
            stats["reconstructable"] += 1
            sys_msg, user_msg, asst_msg = r
            kern = md.get("kernel", "?")
            meta = {"kernel": kern, "model": md.get("model"),
                    "cycles_send": md.get("cycles_send"), "run_dir": md.get("run_dir")}

            if kern in holdout:
                stats["heldout_records"] += 1
                stats["per_kernel_heldout"][kern] = stats["per_kernel_heldout"].get(kern, 0) + 1
                heldout_rows.append({
                    "kernel": kern,
                    "prompt": [sys_msg, user_msg],
                    "reference_csl": asst_msg["content"],  # the commercial-model artifact
                    "model": md.get("model"), "cycles_send": md.get("cycles_send"),
                    "run_dir": md.get("run_dir"),
                })
                continue

            stats["train_pool"] += 1
            # dedup on (user prompt, completion)
            h = hashlib.sha1((user_msg["content"] + "\x00" + asst_msg["content"]).encode()).hexdigest()
            if h in seen:
                stats["dropped_dup"] += 1
                continue
            seen.add(h)

            if tok is not None:
                nt = n_tokens([sys_msg, user_msg], [asst_msg])
                if nt > args.max_seq_len:
                    stats["dropped_over_len"] += 1
                    continue
                tok_lens.append(nt)

            kept.append(([sys_msg, user_msg], [asst_msg], meta))
            stats["per_kernel_kept"][kern] = stats["per_kernel_kept"].get(kern, 0) + 1

    stats["kept"] = len(kept)

    # deterministic split
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_val = int(len(kept) * args.val_frac)
    val, train = kept[:n_val], kept[n_val:]
    stats["train"], stats["val"] = len(train), len(val)

    def dump(split_name, rows):
        with (out / f"{split_name}.jsonl").open("w") as g, \
             (out / f"{split_name}.meta.jsonl").open("w") as gm:
            for p_msgs, c_msgs, meta in rows:
                g.write(json.dumps({"prompt": p_msgs, "completion": c_msgs}) + "\n")
                gm.write(json.dumps(meta) + "\n")

    dump("train", train)
    dump("val", val)
    with (out / "test_prompts.jsonl").open("w") as g:
        for row in heldout_rows:
            g.write(json.dumps(row) + "\n")

    if tok_lens:
        tok_lens.sort()
        n = len(tok_lens)
        stats["token_len"] = {
            "n": n, "min": tok_lens[0],
            "p50": tok_lens[n // 2], "p90": tok_lens[min(n - 1, int(0.9 * n))],
            "max": tok_lens[-1],
        }

    with (out / "prep_summary.json").open("w") as g:
        json.dump(stats, g, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nWROTE: {out}/train.jsonl ({stats['train']}), val.jsonl ({stats['val']}), "
          f"test_prompts.jsonl ({len(heldout_rows)} held-out records)")


if __name__ == "__main__":
    main()
