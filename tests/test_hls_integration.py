"""Integration tests that need a real vitis_hls on PATH (source env_vitis.sh first).

Run: python -m pytest tests/test_hls_integration.py -v
Each test synthesizes for real (~1-4 min); the suite is intentionally small.
"""

from __future__ import annotations

import shutil

import pytest

from hlsenv.runner import (
    STATUS_COMPILE_FAIL,
    STATUS_CSIM_FAIL,
    STATUS_OK,
    STATUS_TIMEOUT,
    run_synthesis,
)
from hlsenv.tasks import load_task

pytestmark = pytest.mark.skipif(
    shutil.which("vitis_hls") is None,
    reason="vitis_hls not on PATH (source scripts/env/env_vitis.sh)",
)

TASK_DIR = "/eagle/argonne_tpc/lyb/projects/llm-hls/kernels/fir"


@pytest.fixture(scope="module")
def task():
    return load_task(TASK_DIR)


def _synth(task, kernel_src, timeout_s=600):
    return run_synthesis(
        kernel_src, task["tb_src"], task["top"],
        part=task["part"], clock_period_ns=task["clock_period_ns"],
        timeout_s=timeout_s, csim=task.get("csim", True),
    )


def test_reference_kernel_synthesizes(task):
    res = _synth(task, task["ref_src"])
    assert res.status == STATUS_OK, res.log_tail
    assert res.qor["lat_worst"] > 0
    assert res.qor["lut"] > 0


def test_broken_syntax_is_compile_fail(task):
    res = _synth(task, "void fir(const int x[128], const int c[8], int y[128]) { this is not C++ }")
    assert res.status == STATUS_COMPILE_FAIL, res.log_tail


def test_wrong_logic_is_csim_fail(task):
    wrong = """
#define N 128
#define TAPS 8
void fir(const int x[N], const int c[TAPS], int y[N]) {
  for (int i = 0; i < N; i++) y[i] = x[i];  // ignores coefficients
}
"""
    res = _synth(task, wrong)
    assert res.status == STATUS_CSIM_FAIL, res.log_tail


def test_csim_loop_bomb_is_killed(task):
    bomb = """
#define N 128
#define TAPS 8
void fir(const int x[N], const int c[TAPS], int y[N]) {
  volatile int spin = 1;
  while (spin) { }   // hangs csim forever
  for (int i = 0; i < N; i++) y[i] = 0;
}
"""
    res = _synth(task, bomb, timeout_s=120)
    assert res.status == STATUS_TIMEOUT
