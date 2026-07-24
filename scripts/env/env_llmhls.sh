# env_llmhls.sh — source from login OR compute nodes to enter the LLM+HLS python env.
# Usage: source /eagle/argonne_tpc/lyb/projects/llm-hls/scripts/env/env_llmhls.sh

EAGLE=/eagle/argonne_tpc/lyb

# ~/miniconda3 base auto-activates in .bashrc and shadows every other python;
# deactivate it fully, then scrub any leftover conda paths from PATH.
if command -v conda >/dev/null 2>&1; then
  while [ -n "${CONDA_PREFIX:-}" ]; do conda deactivate 2>/dev/null || break; done
fi
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v "^$HOME/miniconda3" | paste -sd:)
export PATH
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV

source "$EAGLE/envs/llmhls/bin/activate"

export HF_HOME=$EAGLE/caches/hf
export PIP_CACHE_DIR=$EAGLE/caches/pip
export VLLM_CACHE_ROOT=$EAGLE/caches/vllm
export TMPDIR=$EAGLE/caches/tmp

# Compute nodes have no direct egress — route through the ALCF proxy.
# Login nodes have direct outbound HTTPS; leave them proxy-free.
case "$(hostname)" in
  *login*) : ;;
  *)
    export http_proxy=http://proxy.alcf.anl.gov:3128
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
    export no_proxy="localhost,127.0.0.1,0.0.0.0,*.alcf.anl.gov,*.cm.polaris.alcf.anl.gov,polaris-*"
    ;;
esac
