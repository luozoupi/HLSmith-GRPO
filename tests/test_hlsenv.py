"""Unit tests for hlsenv. Everything here runs WITHOUT a Vitis install:
runner classification/timeout behavior is exercised through a fake `vitis_hls`
shim placed on PATH. Real-Vitis integration tests live in test_hls_integration.py.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hlsenv.extract import extract_cpp
from hlsenv.qor import parse_csynth_xml
from hlsenv.reward import (
    R_BANNED,
    R_OK_FLOOR,
    R_SYNTH_OK_BASE,
    STATUS_REWARDS,
    contains_banned,
    evaluate_completion,
    partition_for_rank,
    shaped_reward,
)
from hlsenv.runner import (
    STATUS_COMPILE_FAIL,
    STATUS_CSIM_FAIL,
    STATUS_NO_CODE,
    STATUS_OK,
    STATUS_SYNTH_FAIL,
    STATUS_TIMEOUT,
    parse_core_range,
    run_synthesis,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- qor parsing

def test_parse_csynth_xml():
    qor = parse_csynth_xml(FIXTURES / "csynth_sample.xml")
    assert qor["lat_worst"] == 129
    assert qor["lat_best"] == 129
    assert qor["ii_max"] == 130
    assert qor["lut"] == 789
    assert qor["ff"] == 456
    assert qor["dsp"] == 3
    assert qor["bram"] == 0
    assert qor["clk_est_ns"] == pytest.approx(7.3)
    assert qor["avail_lut"] == 53200


def test_parse_csynth_xml_undef_latency():
    qor = parse_csynth_xml(FIXTURES / "csynth_undef.xml")
    assert qor["lat_worst"] == -1
    assert qor["ii_max"] == -1
    assert qor["lut"] == 200


# ------------------------------------------------------------------- extract

def test_extract_fenced_cpp():
    text = "Here you go:\n```cpp\nvoid fir(int x) { }\n```\nEnjoy!"
    assert extract_cpp(text) == "void fir(int x) { }\n"


def test_extract_prefers_last_code_block():
    text = "```\nnot really code\n```\n```c++\n#include <cstdio>\nvoid f() {}\n```"
    assert "#include <cstdio>" in extract_cpp(text)


def test_extract_unfenced_code():
    text = '#include <cstdint>\nvoid top(int a[8]) { a[0] = 1; }'
    assert extract_cpp(text).startswith("#include")


def test_extract_prose_returns_none():
    assert extract_cpp("I cannot help with that request.") is None
    assert extract_cpp("") is None
    assert extract_cpp(None) is None


def test_extract_skips_foreign_language_block():
    # A python block before the cpp block must not confuse fence pairing.
    text = ("Reference:\n```python\ndef fir(x): pass\n```\n"
            "And the HLS C++:\n```cpp\n#include <cstdint>\nvoid fir(int *x) { }\n```")
    out = extract_cpp(text)
    assert "#include <cstdint>" in out
    assert "def fir" not in out
    assert "And the HLS" not in out


def test_extract_case_insensitive_tag():
    out = extract_cpp("```Cpp\nvoid fir(int *x) { }\n```")
    assert out == "void fir(int *x) { }\n"
    assert "```" not in out


# -------------------------------------------------------------------- reward

BASELINE = {"lat_worst": 1000}
BUDGETS = {"lut": 1000, "ff": 2000, "dsp": 10, "bram": 10}


def _qor(lat=1000, lut=100, ff=100, dsp=1, bram=0):
    return {"lat_worst": lat, "lut": lut, "ff": ff, "dsp": dsp, "bram": bram}


def test_reward_ladder_ordering():
    assert STATUS_REWARDS[STATUS_NO_CODE] < STATUS_REWARDS[STATUS_COMPILE_FAIL]
    assert STATUS_REWARDS[STATUS_COMPILE_FAIL] <= STATUS_REWARDS[STATUS_SYNTH_FAIL]
    assert STATUS_REWARDS[STATUS_SYNTH_FAIL] < STATUS_REWARDS[STATUS_CSIM_FAIL]
    assert STATUS_REWARDS[STATUS_CSIM_FAIL] < 0
    # any OK design beats every failure mode
    assert shaped_reward(_qor(), BASELINE, BUDGETS) > max(STATUS_REWARDS.values())
    # ... even a grotesquely over-budget one (floor above CSIM_FAIL)
    worst_ok = shaped_reward(_qor(lut=10**6, dsp=10**4), BASELINE, BUDGETS)
    assert worst_ok == pytest.approx(R_OK_FLOOR)
    assert worst_ok > STATUS_REWARDS[STATUS_CSIM_FAIL]


def test_banned_token_detection():
    hack = "```cpp\nvoid fir(int *x) {\n#ifndef __SYNTHESIS__\n  real_work(x);\n#endif\n}\n```"
    assert contains_banned("#ifndef __SYNTHESIS__")
    assert not contains_banned("void fir(int *x) { }")
    # evaluate_completion short-circuits banned code before any synthesis
    assert evaluate_completion({"name": "t"}, hack, None, None) == R_BANNED


def test_degenerate_detection():
    from hlsenv.qor import is_degenerate
    base = {"lat_worst": 72060, "dsp": 12, "lut": 2082}
    # empty circuit: tile>dims => 0-iteration compute (real bicg artifact)
    assert is_degenerate({"lat_worst": 118, "dsp": 0, "lut": 55, "ff": 9})
    # fully empty
    assert is_degenerate({"lat_worst": 0, "dsp": 0, "lut": 0, "ff": 0})
    # collapsed vs a DSP-using baseline
    assert is_degenerate({"lat_worst": 100, "dsp": 0, "lut": 600, "ff": 200}, base)
    # NOT degenerate: constant-coefficient stencil is legitimately DSP-free
    assert not is_degenerate({"lat_worst": 1406, "dsp": 0, "lut": 77547, "ff": 5000})
    # NOT degenerate: normal design
    assert not is_degenerate({"lat_worst": 8312, "dsp": 33, "lut": 16181, "ff": 8000})


def test_reward_monotonic_in_speedup():
    slow = shaped_reward(_qor(lat=2000), BASELINE, BUDGETS)
    par = shaped_reward(_qor(lat=1000), BASELINE, BUDGETS)
    fast = shaped_reward(_qor(lat=500), BASELINE, BUDGETS)
    assert slow < par < fast


def test_reward_speedup_logscale_and_capped():
    # log2 speedup: +0.6 per doubling vs baseline (lat_worst=1000).
    copy = shaped_reward(_qor(lat=1000), BASELINE, BUDGETS)    # 1x  -> +0.2
    x2 = shaped_reward(_qor(lat=500), BASELINE, BUDGETS)       # 2x  -> +0.8
    x4 = shaped_reward(_qor(lat=250), BASELINE, BUDGETS)       # 4x  -> +1.4
    assert copy == pytest.approx(0.2)
    assert x2 == pytest.approx(0.8)
    assert x4 == pytest.approx(1.4)
    # optimization is monotonically rewarded (unlike the old 2x-capped linear term)
    assert copy < x2 < x4
    # ...but the speedup RATIO is capped at 8x, so 8x and 1000x score identically
    x8 = shaped_reward(_qor(lat=125), BASELINE, BUDGETS)       # 8x  -> +2.0 (cap)
    huge = shaped_reward(_qor(lat=1), BASELINE, BUDGETS)       # 1000x -> capped
    assert x8 == pytest.approx(huge)
    assert huge <= 2.0


def test_reward_budget_overage_penalized():
    within = shaped_reward(_qor(lut=999), BASELINE, BUDGETS)
    over = shaped_reward(_qor(lut=2000), BASELINE, BUDGETS)
    assert over < within


def test_reward_undef_latency_neutral():
    assert shaped_reward(_qor(lat=-1), BASELINE, BUDGETS) == 0.0


def test_reward_missing_baseline_lat_gives_base_only():
    r = shaped_reward(_qor(lat=100), {"lat_worst": -1}, BUDGETS)
    assert r == pytest.approx(R_SYNTH_OK_BASE)


# ----------------------------------------------------------------- core range

def test_parse_core_range():
    assert parse_core_range("12-15") == [12, 13, 14, 15]
    assert parse_core_range("1,3-5") == [1, 3, 4, 5]


def test_partition_for_rank():
    cores = list(range(12, 32))  # 20 cores
    seen = []
    for rank in range(4):
        workers, rank_cores = partition_for_rank(16, cores, rank, 4)
        assert workers == 4
        assert len(rank_cores) == 5
        seen.extend(rank_cores)
    assert sorted(seen) == cores  # disjoint cover, no overlap

    # single-process launch keeps everything
    workers, rank_cores = partition_for_rank(16, cores, 0, 1)
    assert workers == 16 and rank_cores == cores

    # no cores specified
    workers, rank_cores = partition_for_rank(16, None, 2, 4)
    assert workers == 4 and rank_cores is None


# ------------------------------------------------- runner (fake vitis_hls shim)

KERNEL = "void f() {}\n"
TB = "int main() { return 0; }\n"


def _install_shim(tmp_path: Path, monkeypatch, body: str) -> None:
    """Put a fake `vitis_hls` on PATH that emulates a given outcome."""
    shim = tmp_path / "bin" / "vitis_hls"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text("#!/bin/bash\n" + body)
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{shim.parent}:{os.environ['PATH']}")


SHIM_OK = """
mkdir -p proj/sol1/syn/report
cp {fixture} proj/sol1/syn/report/csynth.xml
echo "INFO: csynth done"
"""

SHIM_CSIM_COMPILE = """
echo "ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s)."
exit 1
"""

SHIM_CSIM_WRONG = """
echo "ERROR: [SIM 211-100] 'csim_design' failed: nonzero return value."
exit 1
"""

SHIM_SYNTH_FAIL = """
echo "ERROR: [HLS 214-256] synthesizability check failed"
exit 1
"""

SHIM_HANG = """
sleep 300
"""

SHIM_TRUNCATED_XML = """
mkdir -p proj/sol1/syn/report
head -c 200 {fixture} > proj/sol1/syn/report/csynth.xml
echo "INFO: csynth done (but the report write was cut short)"
"""


def _run(tmp_path):
    return run_synthesis(KERNEL, TB, "f", scratch_root=tmp_path / "scratch", timeout_s=5)


def test_runner_ok(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch,
                  SHIM_OK.format(fixture=FIXTURES / "csynth_sample.xml"))
    res = _run(tmp_path)
    assert res.status == STATUS_OK
    assert res.qor["lat_worst"] == 129


def test_runner_compile_fail(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch, SHIM_CSIM_COMPILE)
    assert _run(tmp_path).status == STATUS_COMPILE_FAIL


def test_runner_csim_fail(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch, SHIM_CSIM_WRONG)
    assert _run(tmp_path).status == STATUS_CSIM_FAIL


def test_runner_synth_fail(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch, SHIM_SYNTH_FAIL)
    assert _run(tmp_path).status == STATUS_SYNTH_FAIL


def test_runner_truncated_xml_is_synth_fail_not_crash(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch,
                  SHIM_TRUNCATED_XML.format(fixture=FIXTURES / "csynth_sample.xml"))
    res = _run(tmp_path)
    assert res.status == STATUS_SYNTH_FAIL
    assert res.qor is None


def test_runner_timeout_killed_quickly(tmp_path, monkeypatch):
    import time
    _install_shim(tmp_path, monkeypatch, SHIM_HANG)
    t0 = time.monotonic()
    res = run_synthesis(KERNEL, TB, "f", scratch_root=tmp_path / "scratch", timeout_s=2)
    assert res.status == STATUS_TIMEOUT
    assert time.monotonic() - t0 < 30  # killed at ~2s, not after sleep 300


def test_runner_cleans_scratch(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch,
                  SHIM_OK.format(fixture=FIXTURES / "csynth_sample.xml"))
    scratch = tmp_path / "scratch"
    _run(tmp_path)
    assert not any(scratch.iterdir()) if scratch.exists() else True


def test_runner_materializes_aux_files(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch,
                  'test -f gemm.h && test -f support/common/mc.h || exit 3\n'
                  + SHIM_OK.format(fixture=FIXTURES / "csynth_sample.xml"))
    res = run_synthesis(
        KERNEL, TB, "f", scratch_root=tmp_path / "scratch", timeout_s=5,
        aux_files={"gemm.h": "#define N 8\n", "support/common/mc.h": "// mc\n"},
    )
    assert res.status == STATUS_OK  # shim exits 3 (-> SYNTH_FAIL) if aux missing


def test_runner_rejects_escaping_aux_paths(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch,
                  SHIM_OK.format(fixture=FIXTURES / "csynth_sample.xml"))
    with pytest.raises(ValueError):
        run_synthesis(KERNEL, TB, "f", scratch_root=tmp_path / "scratch",
                      timeout_s=5, aux_files={"../evil.h": "x"})


def test_runner_archives(tmp_path, monkeypatch):
    _install_shim(tmp_path, monkeypatch,
                  SHIM_OK.format(fixture=FIXTURES / "csynth_sample.xml"))
    archive = tmp_path / "archive"
    res = run_synthesis(KERNEL, TB, "f", scratch_root=tmp_path / "scratch",
                        timeout_s=5, archive_dir=archive)
    ep = archive / res.episode_id
    assert (ep / "result.json").exists()
    assert (ep / "kernel.cpp").exists()
    assert (ep / "csynth.xml").exists()
