"""hlsenv — Vitis HLS synthesis as an RL reward environment for TRL GRPO."""

from .extract import extract_cpp
from .qor import parse_csynth_xml
from .reward import HLSRewardFunc, evaluate_completion, shaped_reward
from .runner import SynthResult, run_synthesis
from .tasks import load_task, load_tasks

__all__ = [
    "HLSRewardFunc",
    "SynthResult",
    "evaluate_completion",
    "extract_cpp",
    "load_task",
    "load_tasks",
    "parse_csynth_xml",
    "run_synthesis",
    "shaped_reward",
]
