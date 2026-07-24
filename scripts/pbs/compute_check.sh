#!/bin/bash
# M4 compute-node validation — run inside an interactive debug job:
#   qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A argonne_tpc
#   bash /eagle/argonne_tpc/lyb/projects/llm-hls/scripts/pbs/compute_check.sh
# Batches every check into one node-hour.

set -u
PROJ=/eagle/argonne_tpc/lyb/projects/llm-hls
PASS=0; FAIL=0
ok()   { echo "PASS: $*"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $*"; FAIL=$((FAIL+1)); }

echo "===== 1. GPU sanity ====="
source "$PROJ/scripts/env/env_llmhls.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
python - <<'EOF' && ok "torch sees GPUs" || bad "torch GPU check"
import torch
n = torch.cuda.device_count()
print(f"{n}x {torch.cuda.get_device_name(0)} | wheel CUDA {torch.version.cuda} | archs {torch.cuda.get_arch_list()}")
assert n == 4, f"expected 4 GPUs, got {n}"
assert "sm_80" in torch.cuda.get_arch_list(), "A100 sm80 missing from this torch wheel!"
x = torch.randn(1024, 1024, device="cuda")
print("matmul:", (x @ x).sum().item())
EOF

echo "===== 2. node-local NVMe ====="
df -h /local/scratch /tmp 2>/dev/null
SCR=/local/scratch/${PBS_JOBID%%.*}
mkdir -p "$SCR" && touch "$SCR/.w" && rm "$SCR/.w" && ok "NVMe writable at $SCR" || bad "NVMe scratch"

echo "===== 2b. thread/pid limits (login nodes cap at 256 — jobs must not) ====="
CG=$(cut -d: -f3 /proc/self/cgroup | head -1)
PMAX=$(cat /sys/fs/cgroup"$CG"/pids.max 2>/dev/null || echo unknown)
echo "cgroup pids.max: $PMAX | ulimit -u: $(ulimit -u)"
if [ "$PMAX" = "max" ] || [ "${PMAX:-0}" -gt 4096 ] 2>/dev/null; then
  ok "pid limit fine for 16 parallel vitis_hls + vLLM"
else
  bad "pid limit low ($PMAX) — reduce HLSENV_WORKERS and vLLM threads"
fi

echo "===== 3. Vitis libs present on compute image ====="
MISSING=""
for so in libtinfo.so.5 libncurses.so.5 libX11.so.6 libXext.so.6 libXrender.so.1 libXtst.so.6 libfreetype.so.6 libfontconfig.so.1 libstdc++.so.6; do
  if /sbin/ldconfig -p 2>/dev/null | grep -q "$so"; then echo "  ok  $so"; else echo "  MISSING $so"; MISSING="$MISSING $so"; fi
done
if [ -n "$MISSING" ]; then
  bad "libs missing:$MISSING — staging copies from login node needed (compat_libs)"
else
  ok "all Vitis libs present"
fi

echo "===== 4. vitis_hls on compute node ====="
source "$PROJ/scripts/env/env_vitis.sh"
if which vitis_hls >/dev/null 2>&1 && cp -r "$PROJ/kernels/fir" "$SCR/fir_check" && cd "$SCR/fir_check"; then
  cat > synth.tcl <<'TCL'
set_param general.maxThreads 1
open_project -reset proj
add_files kernel.cpp
add_files -tb tb.cpp
set_top fir
open_solution -reset sol1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
TCL
  if timeout 600 vitis_hls -f synth.tcl > synth_log.txt 2>&1 && [ -f proj/sol1/syn/report/csynth.xml ]; then
    ok "HLS_COMPUTE_OK — real synthesis works on the compute node"
  else
    bad "vitis_hls synthesis failed — see $SCR/fir_check/synth_log.txt (last 30 lines follow)"
    tail -30 synth_log.txt
  fi
  cd - >/dev/null
else
  bad "vitis_hls not on PATH (is M2 done? env_vitis.sh paths right?) or staging to $SCR failed"
fi

echo "===== 5. parallel synthesis calibration ====="
if which vitis_hls >/dev/null 2>&1 && [ -d "$SCR/fir_check/proj" ]; then
  for NW in 8 16 24; do
    rm -rf "$SCR/cal"; mkdir -p "$SCR/cal"
    t0=$SECONDS
    for i in $(seq 1 $NW); do
      ( cp -r "$PROJ/kernels/fir" "$SCR/cal/w$i" && cd "$SCR/cal/w$i" &&
        cp "$SCR/fir_check/synth.tcl" . && vitis_hls -f synth.tcl >/dev/null 2>&1 ) &
    done
    wait
    n_ok=$(ls -d "$SCR"/cal/w*/proj/sol1/syn/report/csynth.xml 2>/dev/null | wc -l)
    echo "  $NW workers: ${n_ok}/$NW succeeded in $((SECONDS-t0))s"
  done
  ok "calibration done — pick HLSENV_WORKERS from the table above"
fi

echo "===== 6. proxy egress from compute node ====="
code=$(curl -s --max-time 15 -o /dev/null -w "%{http_code}" https://pypi.org) || code=000
[ "$code" = "200" ] && ok "proxy egress works (pypi $code)" || bad "proxy egress (got $code)"

echo
echo "===== SUMMARY: $PASS pass, $FAIL fail ====="
exit $((FAIL > 0))
