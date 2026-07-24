# env_vitis.sh — source to put Vitis HLS 2023.2 on PATH (login or compute nodes).
# Usage: source /eagle/argonne_tpc/lyb/projects/llm-hls/scripts/env/env_vitis.sh

XIL=/eagle/argonne_tpc/lyb/tools/xilinx

# The 2023.2 unified installer puts classic Vitis HLS in its own sibling tree
# (Vitis_HLS/2023.2) whose bin dir Vivado's settings64.sh does NOT add — source
# every settings64.sh that exists (Vivado for vivado, Vitis_HLS for vitis_hls).
for _s in "$XIL/Vivado/2023.2/settings64.sh" "$XIL/2023.2/Vivado/settings64.sh" \
          "$XIL/Vitis_HLS/2023.2/settings64.sh" "$XIL/2023.2/Vitis_HLS/settings64.sh"; do
  if [ -f "$_s" ]; then
    source "$_s"
  fi
done
unset _s

# Don't write per-run registry files into $HOME/.Xilinx from parallel workers.
export XILINX_LOCAL_USER_DATA=no

# Compat libs staged here only if a node is missing e.g. libtinfo.so.5
# (populated during M4 compute-node validation if needed).
if [ -d "$XIL/compat_libs" ]; then
  export LD_LIBRARY_PATH="$XIL/compat_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
