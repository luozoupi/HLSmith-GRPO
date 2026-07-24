"""Convert agentic_trl_chat_v1 rows (Gemma-era multi-system chat) to Qwen-compatible
messages for TRL SFT.

Fixes applied per row:
  - leading consecutive `system` turns merged into ONE system message
  - later `system` turns (phase status lines) demoted to `user` turns
  - consecutive same-role turns merged (chat templates require alternation)
  - rows longer than --max-chars dropped (context budget guard)

Usage:
  python convert_rl_corpus.py --src .../agentic_trl_chat_v1 --out .../sft_qwen_v3 \
      [--max-chars 60000] [--min-tier 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fix_messages(messages: list[dict]) -> list[dict]:
    # 1. merge leading systems
    head = []
    i = 0
    while i < len(messages) and messages[i]["role"] == "system":
        head.append(messages[i]["content"])
        i += 1
    fixed = [{"role": "system", "content": "\n\n".join(head)}] if head else []
    # 2. demote later systems to user
    for m in messages[i:]:
        role = "user" if m["role"] == "system" else m["role"]
        fixed.append({"role": role, "content": m["content"]})
    # 3. merge consecutive same-role turns
    merged: list[dict] = []
    for m in fixed:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


def convert(src: Path, out: Path, max_chars: int, min_tier: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        fin = src / f"{split}.jsonl"
        if not fin.exists():
            continue
        kept = dropped_len = dropped_tier = 0
        with fin.open() as f, (out / f"{split}.jsonl").open("w") as g:
            for line in f:
                r = json.loads(line)
                tier = (r.get("metadata") or {}).get("correctness_tier") or 0
                if tier < min_tier:
                    dropped_tier += 1
                    continue
                msgs = fix_messages(r["messages"])
                if not msgs or msgs[-1]["role"] != "assistant":
                    dropped_len += 1
                    continue
                if sum(len(m["content"]) for m in msgs) > max_chars:
                    dropped_len += 1
                    continue
                g.write(json.dumps({
                    "messages": msgs,
                    "benchmark": r.get("benchmark"),
                    "teacher": r.get("model_teacher"),
                    "correctness_tier": tier,
                    "reward_scalar": (r.get("metadata") or {}).get("reward_scalar"),
                }) + "\n")
                kept += 1
        print(f"{split}: kept {kept}, dropped {dropped_len} (shape/length) + {dropped_tier} (tier<{min_tier})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=60000)
    ap.add_argument("--min-tier", type=int, default=2)
    args = ap.parse_args()
    convert(Path(args.src), Path(args.out), args.max_chars, args.min_tier)
