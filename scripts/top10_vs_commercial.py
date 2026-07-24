import sys
from pathlib import Path

sys.path.insert(0, "/eagle/argonne_tpc/lyb/projects/llm-hls/scripts")
from compare_arms import parse_single, parse_multistep, parse_commercial

ROOT = "/eagle/argonne_tpc/lyb/runs"
models = {
    "7B+GRPO (single-shot)": parse_single(f"{ROOT}/eval26/7b_grpo_base"),
    "7B+GRPO (multistep)": parse_multistep(f"{ROOT}/eval26_multistep/7b_grpo_base"),
    "7B+step-GRPO (multistep, PARTIAL)": parse_multistep(f"{ROOT}/eval26_multistep/7b_step_grpo"),
}
comm = parse_commercial("/home/lyb/schema_records.jsonl")

def pretty_arm(arm):
    """enh__opus__flash__all_positive -> 'Opus (flash, all-skills)';
    None/unlabeled -> 'GT baseline (HLSFactory, human)'."""
    if arm == "unlabeled":
        return "GT baseline (HLSFactory)"
    parts = arm.replace("enh__", "").split("__")
    model = {"opus": "Opus", "sonnet": "Sonnet"}.get(parts[0], parts[0])
    rest = ", ".join(parts[1:]) if len(parts) > 1 else ""
    return f"{model} ({rest})" if rest else model


# best commercial cosim cycles per benchmark + which arm achieved it
best_comm = {}
for arm, benches in comm.items():
    for b, c in benches.items():
        if c["cosim"] and c.get("cyc_cosim"):
            if b not in best_comm or c["cyc_cosim"] < best_comm[b][1]:
                best_comm[b] = (pretty_arm(arm), c["cyc_cosim"])

for name, arm in models.items():
    rows = [(b, v["cyc_cosim"]) for b, v in arm.items() if v["cosim"] and v["cyc_cosim"]]
    rows.sort(key=lambda x: x[1])  # fastest (best QoR) first
    print(f"\n### {name} — top 10 cosim-verified kernels vs BEST arm on same benchmark")
    print(f"{'benchmark':16s} {'7B cyc':>10s}   {'best arm on this bench':36s} {'their cyc':>10s} {'7B/them':>8s}")
    for b, cyc in rows[:10]:
        bc = best_comm.get(b)
        if bc:
            arm_n, ccyc = bc
            print(f"{b:16s} {cyc:>10,}   {arm_n:36s} {ccyc:>10,} {cyc / ccyc:>7.2f}x")
        else:
            print(f"{b:16s} {cyc:>10,}   {'(no arm cosim-passed)':36s} {'-':>10s} {'-':>8s}")
    wins = sum(1 for b, cyc in rows[:10] if best_comm.get(b) and cyc <= best_comm[b][1])
    print(f"  cosim-verified total: {len(rows)} | of top 10, 7B <= best commercial on: {wins}")
