"""Shaped QoR reward + TRL GRPOTrainer-compatible batched reward function.

Reward ladder (csim gate is mandatory for any positive reward — anti reward-hacking):
    NO_CODE / BANNED  -1.0   no extractable C++, or uses a banned escape hatch
    COMPILE_FAIL      -0.8   csim couldn't compile the kernel
    SYNTH_FAIL        -0.8   compiled+simulated but csynth_design failed
    TIMEOUT           -0.8   synthesis killed at the per-episode time budget
    CSIM_FAIL         -0.6   compiled, but wrong results vs the golden testbench
    OK                shaped in [-0.4, 2]: base + capped speedup vs baseline
                      − resource overage; floored ABOVE every failure mode so a
                      correct design always beats an incorrect one.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

from .extract import extract_cpp
from .qor import is_degenerate
from .runner import (
    STATUS_COMPILE_FAIL,
    STATUS_CSIM_FAIL,
    STATUS_NO_CODE,
    STATUS_OK,
    STATUS_SYNTH_FAIL,
    STATUS_TIMEOUT,
    parse_core_range,
    pin_to_cores,
    run_synthesis,
)
from .tasks import load_tasks

STATUS_REWARDS = {
    STATUS_NO_CODE: -1.0,
    STATUS_COMPILE_FAIL: -0.8,
    STATUS_SYNTH_FAIL: -0.8,
    STATUS_TIMEOUT: -0.8,
    STATUS_CSIM_FAIL: -0.6,
}

R_BANNED = -1.0
# A design that synthesizes to an empty/do-nothing circuit (e.g. a tile size
# larger than the problem dims → zero-iteration compute loop) can pass a weak
# testbench and collect a huge fake "speedup". Penalize like a wrong result.
R_DEGENERATE = -0.6
# __SYNTHESIS__ lets code behave correctly in csim while synthesizing different
# (e.g. empty) hardware — the canonical way to fool a csim-gated QoR reward.
BANNED_TOKENS = ("__SYNTHESIS__",)

R_SYNTH_OK_BASE = 0.2
# Floor for csim-PASSING designs. Must stay above CSIM_FAIL (-0.6) so a correct
# over-budget design is never ranked below wrong-results code.
R_OK_FLOOR = -0.4
SPEEDUP_WEIGHT = 0.6
SPEEDUP_CAP = 2.0
OVERAGE_WEIGHT = 0.5
BUDGET_KEYS = ("lut", "ff", "dsp", "bram")


def contains_banned(code: str) -> bool:
    return any(tok in code for tok in BANNED_TOKENS)


def shaped_reward(qor: dict, baseline: dict, budgets: dict) -> float:
    """Reward for a design that synthesized AND passed csim."""
    reward = R_SYNTH_OK_BASE

    lat = qor.get("lat_worst", -1)
    base_lat = baseline.get("lat_worst", -1)
    if lat <= 0:
        # 'undef' latency (unbounded loops): synthesized, but QoR is unusable.
        return 0.0
    if base_lat > 0:
        reward += SPEEDUP_WEIGHT * min(SPEEDUP_CAP, base_lat / lat)

    overage = 0.0
    for key in BUDGET_KEYS:
        budget = budgets.get(key)
        used = qor.get(key, -1)
        if budget and used > 0:
            overage += max(0.0, used / budget - 1.0)
    reward -= OVERAGE_WEIGHT * overage

    return max(R_OK_FLOOR, min(2.0, reward))


def evaluate_completion(task: dict, completion_text: str,
                        scratch_root: str | None, archive_dir: str | None) -> float:
    """Extract → synthesize → score one completion. Module-level: picklable for the pool."""
    code = extract_cpp(completion_text)
    if code is None:
        return STATUS_REWARDS[STATUS_NO_CODE]
    if contains_banned(code):
        return R_BANNED

    baseline = task.get("baseline")
    if not baseline:
        raise RuntimeError(
            f"task '{task['name']}' has no baseline QoR — run: python -m hlsenv baseline {task['dir']}"
        )

    try:
        result = run_synthesis(
            code,
            task["tb_src"],
            task["top"],
            part=task["part"],
            clock_period_ns=task["clock_period_ns"],
            scratch_root=scratch_root,
            timeout_s=task.get("timeout_s", 300),
            csim=task.get("csim", True),
            archive_dir=archive_dir,
            aux_files=task.get("aux_files") or None,
        )
    except RuntimeError:
        raise  # missing vitis_hls etc. — a config error, fail fast
    except Exception:
        # One bad episode (disk full, race, parse oddity) must never kill a
        # multi-hour training job — score it as a failed synthesis.
        traceback.print_exc(file=sys.stderr)
        return STATUS_REWARDS[STATUS_SYNTH_FAIL]

    if result.status != STATUS_OK:
        return STATUS_REWARDS[result.status]
    # A csim-passing but empty circuit is reward-hacking — don't pay for it.
    if is_degenerate(result.qor, baseline):
        return R_DEGENERATE
    return shaped_reward(result.qor, baseline, task.get("budgets", {}))


def _completion_text(completion) -> str:
    """TRL passes either plain strings or chat-format [{'role','content'}, ...]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
    return str(completion)


def partition_for_rank(total_workers: int, cores: list[int] | None,
                       rank: int, world: int) -> tuple[int, list[int] | None]:
    """Split the per-node HLS worker/core budget across accelerate ranks.

    TRL calls reward functions on EVERY rank with that rank's completion slice,
    so each rank owns its own pool — without this split, N ranks × N workers
    would oversubscribe the same cores.
    """
    workers = max(1, total_workers // max(1, world))
    rank_cores = cores[rank % world::world] if cores else None
    return workers, (rank_cores or cores)


class HLSRewardFunc:
    """Batched reward callable for TRL GRPOTrainer.

    Dataset rows must carry a `task_id` column; TRL forwards extra columns as
    per-batch kwargs lists. Synthesis fans out over a spawn-based process pool
    pinned away from the GPU/vLLM cores.

    Env knobs: HLSENV_WORKERS = TOTAL workers per node (default 16), divided
    across ranks; HLSENV_CORES (e.g. "12-31"), partitioned across ranks;
    HLSENV_SCRATCH, HLSENV_ARCHIVE.
    """

    __name__ = "hls_reward"

    def __init__(self, kernels_dir: str, *, max_workers: int | None = None,
                 cores: str | None = None, scratch_root: str | None = None,
                 archive_dir: str | None = None):
        self.tasks = load_tasks(kernels_dir)
        total = max_workers or int(os.environ.get("HLSENV_WORKERS", "16"))
        core_spec = cores or os.environ.get("HLSENV_CORES", "")
        all_cores = parse_core_range(core_spec) if core_spec else None
        world = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        self.max_workers, self.cores = partition_for_rank(total, all_cores, rank, world)
        self.scratch_root = scratch_root or os.environ.get("HLSENV_SCRATCH")
        self.archive_dir = archive_dir or os.environ.get("HLSENV_ARCHIVE")
        self._pool: ProcessPoolExecutor | None = None

    def _ensure_pool(self) -> ProcessPoolExecutor:
        # Lazy + spawn: never fork a process that may already hold CUDA state.
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=pin_to_cores,
                initargs=(self.cores,),
            )
        return self._pool

    def __call__(self, prompts=None, completions=None, task_id=None, **kwargs) -> list[float]:
        if completions is None:
            return []
        if task_id is None:
            if len(self.tasks) != 1:
                raise ValueError("dataset must carry a task_id column when multiple tasks exist")
            task_id = [next(iter(self.tasks))] * len(completions)

        pool = self._ensure_pool()
        futures = [
            pool.submit(
                evaluate_completion,
                self.tasks[tid],
                _completion_text(completion),
                self.scratch_root,
                self.archive_dir,
            )
            for tid, completion in zip(task_id, completions)
        ]
        return [f.result() for f in futures]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
