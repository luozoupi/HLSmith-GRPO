# env_vitis.sh — put Vitis HLS on PATH (login or compute nodes).
# Usage: source <repo>/scripts/env/env_vitis.sh
#
# Portable across HPC sites. Set XILINX_ROOT / XILINX_VERSION in scripts/env/site.sh,
# or pre-load a site module (`module load vitis/2023.2`) and this script will detect it.

_HLSMITH_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_HLSMITH_REPO="$(cd "$_HLSMITH_HERE/../.." && pwd)"
[ -f "$_HLSMITH_HERE/site.sh" ] && . "$_HLSMITH_HERE/site.sh"

: "${HLSMITH_ROOT:=$(cd "$_HLSMITH_REPO/../.." && pwd)}"
: "${XILINX_VERSION:=2023.2}"

# Candidate install roots, first match wins.
for _r in "${XILINX_ROOT:-}" "$HLSMITH_ROOT/tools/xilinx" /opt/Xilinx /opt/xilinx \
          /usr/local/Xilinx "$HOME/Xilinx"; do
  [ -n "$_r" ] && [ -d "$_r" ] && { XILINX_ROOT="$_r"; break; }
done

# The unified installer puts classic Vitis HLS in its own sibling tree
# (Vitis_HLS/<ver>) whose bin dir Vivado's settings64.sh does NOT add — source
# every settings64.sh that exists (Vivado for vivado, Vitis_HLS for vitis_hls).
if [ -n "${XILINX_ROOT:-}" ]; then
  for _s in "$XILINX_ROOT/Vivado/$XILINX_VERSION/settings64.sh" \
            "$XILINX_ROOT/$XILINX_VERSION/Vivado/settings64.sh" \
            "$XILINX_ROOT/Vitis_HLS/$XILINX_VERSION/settings64.sh" \
            "$XILINX_ROOT/$XILINX_VERSION/Vitis_HLS/settings64.sh" \
            "$XILINX_ROOT/Vitis/$XILINX_VERSION/settings64.sh"; do
    [ -f "$_s" ] && . "$_s"
  done
  unset _s
fi

# Don't write per-run registry files into $HOME/.Xilinx from parallel workers.
export XILINX_LOCAL_USER_DATA=no

# Classic 2023.2 runs a tcl via `vitis_hls -f`; 2024.1+ uses `vitis-run --tcl --input_file`.
export HLS_TCL_RUNNER="${HLS_TCL_RUNNER:-vitis_hls -f}"

# Compat libs, if a node is missing e.g. libtinfo.so.5 (see REPRODUCE.md troubleshooting).
if [ -n "${XILINX_ROOT:-}" ] && [ -d "$XILINX_ROOT/compat_libs" ]; then
  export LD_LIBRARY_PATH="$XILINX_ROOT/compat_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if ! command -v vitis_hls >/dev/null 2>&1; then
  echo "env_vitis.sh: vitis_hls not on PATH (looked under XILINX_ROOT=${XILINX_ROOT:-unset})" >&2
  echo "  set XILINX_ROOT/XILINX_VERSION in scripts/env/site.sh, or 'module load' Vitis first" >&2
fi

unset _HLSMITH_HERE _HLSMITH_REPO _r
