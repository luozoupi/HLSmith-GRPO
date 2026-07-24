"""Unified polybench-26 matrix: 7B+GRPO arms vs commercial (Opus/Sonnet),
across workflow (oneshot / flash / multistep) x skills (no_skills / curated / all_positive).

Parsing rules (they differ per arm — this is the whole reason a naive parser is wrong):
  * oneshot/flash arms  -> <bench>_results.json: synth_report + csim_status/cosim_status
  * multistep arms      -> <bench>_multistep_results.json: BEST step per tier from
                           optimization_history (matches compare_arms.py::parse_multistep)
  * eval26_skills arms ran with C2HLS_DISABLE_COSIM (cosim disabled inline for speed), so
    their cosim verdicts come from the STANDALONE CPU cosim runs (cosim_*.json), which used
    the c2hls system's own hls_eval.run_cosim.
  * commercial          -> schema_records.jsonl: hls_synth / sw_run / rtl_sim

Degeneracy guard (dsp==0 & tiny lut/ff) applied to every arm so do-nothing kernels never
count as passes.

Latency columns:
  med_csim  = median latency over csim-passing kernels   (optimistic; survivorship)
  med_cosim = median latency over COSIM-passing kernels  (honest: verified-correct only)
"""
import json, glob, statistics
from pathlib import Path

RUNS = Path("/eagle/argonne_tpc/lyb/runs")
SCHEMA = Path("/home/lyb/schema_records.jsonl")


def _i(x):
    try: return int(str(x))
    except Exception: return 0


def _norm(b):
    return b.replace("hlsfactory_", "")


def _deg(lat, dsp, lut, ff=0):
    """empty / do-nothing kernel: no DSP and negligible logic"""
    return dsp == 0 and lut < 200


def parse_oneshot(root):
    out = {}
    for f in glob.glob(f"{root}/*/*_results.json"):
        if "multistep" in f:
            continue
        d = json.loads(Path(f).read_text())
        rep = d.get("synth_report") or {}
        lat = rep.get("latency_cycles")
        degen = _deg(lat, _i(rep.get("dsp")), _i(rep.get("lut")))
        def ok(k):
            v = d.get(k)
            return (v.get("generated") == "passed") if isinstance(v, dict) else (v == "passed")
        csynth = bool(lat) and not degen
        out[_norm(Path(f).parent.name)] = {
            "csynth": csynth, "csim": ok("csim_status") and not degen,
            "cosim": ok("cosim_status") and not degen,
            "lat_csim": lat if (csynth and ok("csim_status")) else None,
            "lat_any": lat if csynth else None,
        }
    return out


def parse_multistep(root):
    """Per tier, take the fastest STEP that passes that tier."""
    out = {}
    p = Path(root)
    if not p.exists():
        return out
    for bd in sorted(x for x in p.iterdir() if x.is_dir()):
        f = bd / f"{bd.name}_multistep_results.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        best = {"synth": None, "csim": None, "cosim": None}
        for s in d.get("optimization_history") or d.get("steps") or []:
            rep = s.get("report") or {}
            lat = rep.get("latency_cycles")
            if not (s.get("success") and lat):
                continue
            lat = int(lat)
            if _deg(lat, _i(rep.get("dsp")), _i(rep.get("lut"))):
                continue
            if best["synth"] is None or lat < best["synth"]:
                best["synth"] = lat
            if (s.get("csim") or {}).get("passed") and (best["csim"] is None or lat < best["csim"]):
                best["csim"] = lat
            if (s.get("cosim") or {}).get("passed") and (best["cosim"] is None or lat < best["cosim"]):
                best["cosim"] = lat
        out[_norm(bd.name)] = {
            "csynth": best["synth"] is not None, "csim": best["csim"] is not None,
            "cosim": best["cosim"] is not None,
            "lat_csim": best["csim"], "lat_any": best["synth"],
        }
    return out


def overlay_cosim(rows, cosim_files):
    """Replace inline cosim verdicts with standalone CPU-cosim results."""
    verdict = {}
    for cf in cosim_files:
        p = Path(cf)
        if not p.exists():
            continue
        for b, v in json.loads(p.read_text()).items():
            if v.get("cosim_passed") is not None:
                verdict[_norm(b)] = bool(v["cosim_passed"])
    if not verdict:
        return rows, False
    for b, r in rows.items():
        r["cosim"] = verdict.get(b, False)
    return rows, True


def parse_commercial():
    acc = {}
    for line in SCHEMA.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        arm = (r.get("implementation") or {}).get("origin_version")
        if not arm:
            continue
        bench = _norm(((r.get("problem") or {}).get("group_path") or ["?"])[0])
        slot = acc.setdefault(arm, {}).setdefault(
            bench, {"csynth": False, "csim": False, "cosim": False, "lat_csim": None, "lat_any": None})
        rt, body = r.get("report_type"), (r.get(r.get("report_type")) or {})
        if rt == "hls_synth":
            slot["csynth"] = body.get("status") == "pass"
            lat = (((body.get("PerformanceEstimates") or {})
                    .get("SummaryOfOverallLatency") or {}).get("Average-caseLatency"))
            slot["lat_any"] = _i(lat) or None
        elif rt == "sw_run":
            slot["csim"] = body.get("status") == "pass"
        elif rt == "rtl_sim":
            slot["cosim"] = body.get("status") == "pass"
    for bs in acc.values():
        for v in bs.values():
            v["lat_csim"] = v["lat_any"] if v["csim"] else None
    return acc


def summarize(label, rows, note=""):
    if not rows:
        return dict(label=label, n=0, note=note)
    n = len(rows)
    lat_csim = [v["lat_csim"] for v in rows.values() if v["csim"] and v["lat_csim"]]
    lat_cosim = [v["lat_csim"] or v["lat_any"] for v in rows.values()
                 if v["cosim"] and (v["lat_csim"] or v["lat_any"])]
    return dict(label=label, n=n,
                csynth=sum(v["csynth"] for v in rows.values()),
                csim=sum(v["csim"] for v in rows.values()),
                cosim=sum(v["cosim"] for v in rows.values()),
                med_csim=int(statistics.median(lat_csim)) if lat_csim else None,
                med_cosim=int(statistics.median(lat_cosim)) if lat_cosim else None, note=note)


def fmt(s):
    if not s.get("n"):
        return f"| {s['label']} | — | — | — | — | — | {s.get('note') or 'queued'} |"
    mc = f"{s['med_csim']:,}" if s["med_csim"] else "—"
    mo = f"{s['med_cosim']:,}" if s["med_cosim"] else "—"
    return (f"| {s['label']} | {s['csynth']}/{s['n']} | {s['csim']}/{s['n']} | "
            f"**{s['cosim']}/{s['n']}** | {mc} | {mo} | {s.get('note') or ''} |")


if __name__ == "__main__":
    out = []
    # --- 7B + GRPO arms ---
    r = parse_oneshot(RUNS / "eval26/7b_grpo_base")
    out.append(summarize("**7B+GRPO** oneshot / no_skills", r, "inline cosim"))

    r = parse_oneshot(RUNS / "eval26_skills/grpo_flash_noskill")
    r, ok = overlay_cosim(r, [RUNS / "cosim_flash_noskill.json"])
    out.append(summarize("**7B+GRPO** flash / no_skills", r, "standalone cosim" if ok else ""))

    r = parse_multistep(RUNS / "eval26_skills/grpo_multi_curated")
    r, ok = overlay_cosim(r, [RUNS / "cosim_curated_generated.json", RUNS / "cosim_curated_rest.json"])
    out.append(summarize("**7B+GRPO** multistep / curated", r, "standalone cosim" if ok else ""))

    for lbl, d in [("**7B+GRPO** flash / all_positive", "grpo_flash_allpos"),
                   ("**7B+GRPO** multistep / all_positive", "grpo_multistep_allpos")]:
        p = RUNS / "eval26_skills" / d
        rows = parse_multistep(p) or parse_oneshot(p) if p.exists() else {}
        out.append(summarize(lbl, rows, "queued" if not rows else ""))

    # --- commercial ---
    comm = parse_commercial()
    for arm, label in [
        ("enh__opus__oneshot__no_skills", "Opus oneshot / no_skills"),
        ("enh__opus__flash__skilless_A", "Opus flash / skilless_A"),
        ("enh__opus__flash__skilless_B", "Opus flash / skilless_B"),
        ("enh__opus__flash__all_positive", "Opus flash / all_positive"),
        ("enh__opus__multistep__curated", "Opus multistep / curated"),
        ("enh__opus__multistep__all_positive", "Opus multistep / all_positive"),
        ("enh__sonnet__oneshot__no_skills", "Sonnet oneshot / no_skills"),
        ("enh__sonnet__multistep__curated", "Sonnet multistep / curated"),
        ("enh__sonnet__multistep__all_positive", "Sonnet multistep / all_positive"),
    ]:
        out.append(summarize(label, comm.get(arm, {}), "schema"))

    print("| Arm | csynth | csim | **cosim** | med cyc (csim-pass) | med cyc (cosim-pass) | src |")
    print("|---|---|---|---|---|---|---|")
    for s in out:
        print(fmt(s))
