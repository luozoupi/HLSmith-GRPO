"""Run one isolated, time-boxed Vitis HLS synthesis and classify the outcome.

Requires `vitis_hls` (2023.2 classic flow) on PATH — source scripts/env/env_vitis.sh.
Scratch must be node-local (compute: /local/scratch/$PBS_JOBID, login: /tmp) — never Lustre.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import string
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .qor import parse_csynth_xml

STATUS_OK = "OK"
STATUS_NO_CODE = "NO_CODE"
STATUS_COMPILE_FAIL = "COMPILE_FAIL"
STATUS_CSIM_FAIL = "CSIM_FAIL"
STATUS_SYNTH_FAIL = "SYNTH_FAIL"
STATUS_TIMEOUT = "TIMEOUT"

_SYNTH_TCL = string.Template(
    """\
set_param general.maxThreads 1
open_project -reset proj
add_files kernel.cpp
add_files -tb tb.cpp
set_top ${top}
open_solution -reset sol1
set_part {${part}}
create_clock -period ${clock_period} -name default
${csim_line}
csynth_design
exit
"""
)

_CSYNTH_XML_REL = Path("proj/sol1/syn/report/csynth.xml")


@dataclass
class SynthResult:
    status: str
    qor: dict | None
    episode_id: str
    wall_s: float
    log_tail: str


def default_scratch_root() -> Path:
    """Node-local scratch: PBS NVMe when in a job, /tmp otherwise."""
    if os.environ.get("HLSENV_SCRATCH"):
        return Path(os.environ["HLSENV_SCRATCH"])
    jobid = os.environ.get("PBS_JOBID")
    if jobid and Path("/local/scratch").is_dir():
        return Path("/local/scratch") / jobid / "hlsenv"
    return Path("/tmp") / f"hlsenv_{os.environ.get('USER', 'user')}"


def run_synthesis(
    kernel_src: str,
    tb_src: str,
    top: str,
    *,
    part: str = "xc7z020clg400-1",
    clock_period_ns: float = 10,
    scratch_root: str | Path | None = None,
    timeout_s: int = 300,
    csim: bool = True,
    archive_dir: str | Path | None = None,
    keep_scratch: bool = False,
    aux_files: dict[str, str] | None = None,
) -> SynthResult:
    root = Path(scratch_root) if scratch_root else default_scratch_root()
    episode_id = f"ep_{uuid.uuid4().hex[:12]}"
    d = root / episode_id
    d.mkdir(parents=True)

    (d / "kernel.cpp").write_text(kernel_src)
    (d / "tb.cpp").write_text(tb_src)
    # Headers/support files the kernel includes (e.g. gemm.h, support/common/mc.h).
    # vitis_hls finds them via the source dir; no add_files needed.
    for rel, content in (aux_files or {}).items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"aux file path escapes scratch dir: {rel}")
        target = d / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (d / "synth.tcl").write_text(
        _SYNTH_TCL.substitute(
            top=top,
            part=part,
            clock_period=clock_period_ns,
            csim_line="csim_design" if csim else "# csim disabled",
        )
    )

    t0 = time.monotonic()
    status = None
    try:
        try:
            # New session => own process group, so a timeout kills vitis_hls AND
            # every child it spawned (csim gcc, apcc, ...), not just the wrapper.
            proc = subprocess.Popen(
                ["vitis_hls", "-f", "synth.tcl"],
                cwd=d,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "vitis_hls not on PATH — source scripts/env/env_vitis.sh first"
            )
        try:
            out, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, _ = proc.communicate()
            status = STATUS_TIMEOUT
        wall_s = time.monotonic() - t0
        out = out or ""

        qor = None
        xml = d / _CSYNTH_XML_REL
        if status is None:
            if xml.exists():
                # A killed/ENOSPC'd vitis_hls can leave a truncated csynth.xml;
                # score it as a failed synthesis instead of crashing the caller.
                try:
                    qor = parse_csynth_xml(xml)
                    status = STATUS_OK
                except Exception:
                    status = STATUS_SYNTH_FAIL
            elif "'csim_design' failed: compilation" in out or "Compilation errors" in out:
                status = STATUS_COMPILE_FAIL
            elif "'csim_design' failed" in out or "CSim failed" in out:
                status = STATUS_CSIM_FAIL
            else:
                status = STATUS_SYNTH_FAIL

        result = SynthResult(
            status=status,
            qor=qor,
            episode_id=episode_id,
            wall_s=wall_s,
            log_tail=out[-4000:],
        )

        if archive_dir is not None:
            _archive(d, Path(archive_dir) / episode_id, result, out)
        return result
    finally:
        if not keep_scratch:
            shutil.rmtree(d, ignore_errors=True)


def _archive(scratch: Path, dest: Path, result: SynthResult, full_log: str) -> None:
    """Persist the tiny artifacts worth keeping (sources, reports, verdict) to Lustre."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").write_text(json.dumps(asdict(result) | {"log_tail": ""}, indent=2))
    (dest / "vitis_hls.log").write_text(full_log[-100_000:])
    shutil.copy(scratch / "kernel.cpp", dest / "kernel.cpp")
    for rel in (_CSYNTH_XML_REL, Path("proj/sol1/syn/report") / "csynth.rpt"):
        src = scratch / rel
        if src.exists():
            shutil.copy(src, dest / src.name)


def pin_to_cores(cores: list[int] | None) -> None:
    """ProcessPoolExecutor initializer: keep HLS workers off the GPU/vLLM cores."""
    if cores:
        os.sched_setaffinity(0, set(cores))


def parse_core_range(spec: str) -> list[int]:
    """'12-31' or '12-15,24-31' -> [12, 13, ...]."""
    cores: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-")
            cores.extend(range(int(lo), int(hi) + 1))
        elif chunk:
            cores.append(int(chunk))
    return cores
