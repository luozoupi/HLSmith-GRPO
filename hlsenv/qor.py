"""Parse Vitis HLS 2023.2 csynth.xml reports into a flat QoR dict."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _to_int(text: str | None) -> int:
    """'128' -> 128, 'undef'/missing -> -1 (unbounded loops report 'undef')."""
    if text is None:
        return -1
    t = text.strip()
    if not t or "undef" in t.lower():
        return -1
    try:
        return int(float(t))
    except ValueError:
        return -1


def _to_float(text: str | None) -> float:
    if text is None:
        return -1.0
    try:
        return float(text.strip())
    except ValueError:
        return -1.0


def is_degenerate(qor: dict, baseline: dict | None = None) -> bool:
    """True if a design is an empty/do-nothing circuit that a weak testbench may
    vacuously pass (reward-hacking risk). Signatures:
      - zero/negative latency, or essentially no logic (LUT<=100);
      - no arithmetic at all (DSP=0) AND tiny logic (LUT<512, FF<128) — e.g. a
        tile size larger than the problem dims makes the compute loop run 0x;
      - latency collapsed to <2% of a DSP-using baseline while DSP drops to 0.
    NOT flagged: legitimately DSP-free kernels with real logic (e.g. constant-
    coefficient stencils synthesize multiplies into LUTs — jacobi-1d: 77k LUT).
    """
    lat = qor.get("lat_worst", -1)
    lut = qor.get("lut", -1)
    dsp = qor.get("dsp", -1)
    ff = qor.get("ff", -1)
    if lat is not None and lat <= 0:
        return True
    if lut is not None and 0 <= lut <= 100:
        return True
    if dsp == 0 and 0 <= lut < 512 and (ff is None or ff < 128):
        return True
    if baseline:
        blat = baseline.get("lat_worst", -1)
        bdsp = baseline.get("dsp", -1)
        if blat and blat > 0 and lat and lat > 0 and lat < 0.02 * blat and bdsp and bdsp > 0 and dsp == 0:
            return True
    return False


def parse_csynth_xml(path: str | Path) -> dict:
    """Extract latency/II/timing/area from a csynth.xml produced by csynth_design.

    Returns -1 for any field the report omits or marks 'undef'.
    """
    root = ET.parse(str(path)).getroot()

    lat = root.find("PerformanceEstimates/SummaryOfOverallLatency")
    timing = root.find("PerformanceEstimates/SummaryOfTimingAnalysis")
    area = root.find("AreaEstimates/Resources")
    avail = root.find("AreaEstimates/AvailableResources")

    def g(node: ET.Element | None, *names: str) -> int:
        if node is None:
            return -1
        for n in names:
            v = node.findtext(n)
            if v is not None:
                return _to_int(v)
        return -1

    return {
        "lat_best": g(lat, "Best-caseLatency"),
        "lat_avg": g(lat, "Average-caseLatency"),
        "lat_worst": g(lat, "Worst-caseLatency"),
        "ii_min": g(lat, "Interval-min"),
        "ii_max": g(lat, "Interval-max"),
        "clk_est_ns": _to_float(timing.findtext("EstimatedClockPeriod")) if timing is not None else -1.0,
        # DSP48E is the pre-2021 tag name; keep as fallback.
        "lut": g(area, "LUT"),
        "ff": g(area, "FF"),
        "dsp": g(area, "DSP", "DSP48E"),
        "bram": g(area, "BRAM_18K"),
        "uram": g(area, "URAM"),
        "avail_lut": g(avail, "LUT"),
        "avail_ff": g(avail, "FF"),
        "avail_dsp": g(avail, "DSP", "DSP48E"),
        "avail_bram": g(avail, "BRAM_18K"),
    }
