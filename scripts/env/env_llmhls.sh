# env_llmhls.sh — source from login OR compute nodes to enter the LLM+HLS python env.
# Usage: source <repo>/scripts/env/env_llmhls.sh
#
# Portable across HPC sites. Per-machine settings go in scripts/env/site.sh
# (see site.sh.example); without one, paths are inferred from this script's location.

_HLSMITH_HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"   # <repo>/scripts/env
_HLSMITH_REPO="$(cd "$_HLSMITH_HERE/../.." && pwd)"                # <repo>

# Per-machine overrides (untracked).
[ -f "$_HLSMITH_HERE/site.sh" ] && . "$_HLSMITH_HERE/site.sh"

# Default layout: repo lives at $HLSMITH_ROOT/projects/llm-hls, so root is two up.
: "${HLSMITH_ROOT:=$(cd "$_HLSMITH_REPO/../.." && pwd)}"
: "${HLSMITH_VENV:=$HLSMITH_ROOT/envs/llmhls}"
export HLSMITH_ROOT HLSMITH_VENV

# A conda base env that auto-activates in .bashrc shadows the venv's python.
# Deactivate it and scrub its paths. HLSMITH_KEEP_CONDA=1 opts out.
if [ -z "${HLSMITH_KEEP_CONDA:-}" ]; then
  if command -v conda >/dev/null 2>&1; then
    while [ -n "${CONDA_PREFIX:-}" ]; do conda deactivate 2>/dev/null || break; done
  fi
  _HLSMITH_CONDA_BASE="${_HLSMITH_CONDA_BASE:-$HOME/miniconda3}"
  PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v "^$_HLSMITH_CONDA_BASE" | paste -sd:)
  export PATH
  unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
fi

if [ -f "$HLSMITH_VENV/bin/activate" ]; then
  . "$HLSMITH_VENV/bin/activate"
else
  echo "env_llmhls.sh: no virtualenv at $HLSMITH_VENV" >&2
  echo "  create it:  python -m venv $HLSMITH_VENV && pip install -r $_HLSMITH_REPO/requirements.lock" >&2
  echo "  or set HLSMITH_VENV in scripts/env/site.sh" >&2
fi

# Caches on the project filesystem, not $HOME (quota) — override in site.sh.
export HF_HOME="${HF_HOME:-$HLSMITH_ROOT/caches/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HLSMITH_ROOT/caches/pip}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HLSMITH_ROOT/caches/vllm}"
export TMPDIR="${TMPDIR:-$HLSMITH_ROOT/caches/tmp}"

# Login nodes usually run under a tight process/thread cgroup. numpy/OpenBLAS
# default to one thread per core (64+ here), which blows past it: OpenBLAS fails
# thread_init, numpy dies with "CPU dispatcher tracer already initlized", and the
# process SEGFAULTS -- dumping a multi-GB core file. Cap threads outside jobs.
# Inside a batch job (PBS_JOBID/SLURM_JOB_ID set) leave threading untouched.
if [ -z "${PBS_JOBID:-}${SLURM_JOB_ID:-}" ]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
fi

# Some sites give compute nodes no direct egress. Set HLSMITH_PROXY in site.sh
# (empty string = never proxy). Unset => auto-detect ALCF Polaris.
if [ -z "${HLSMITH_PROXY+x}" ]; then
  case "$(hostname -f 2>/dev/null || hostname)" in
    *alcf.anl.gov|*polaris*) HLSMITH_PROXY="http://proxy.alcf.anl.gov:3128" ;;
    *)                       HLSMITH_PROXY="" ;;
  esac
fi
# Login nodes normally have direct egress; only proxy elsewhere.
case "$(hostname)" in
  *login*) : ;;
  *)
    if [ -n "$HLSMITH_PROXY" ]; then
      export http_proxy="$HLSMITH_PROXY"  https_proxy="$HLSMITH_PROXY"
      export HTTP_PROXY="$HLSMITH_PROXY"  HTTPS_PROXY="$HLSMITH_PROXY"
      export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,*.alcf.anl.gov,polaris-*}"
    fi
    ;;
esac

unset _HLSMITH_HERE _HLSMITH_REPO _HLSMITH_CONDA_BASE
