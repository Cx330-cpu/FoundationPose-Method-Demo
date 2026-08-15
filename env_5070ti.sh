#!/usr/bin/env bash
# Source this file before running FoundationPose with the local RTX 5070 Ti env.

_fp_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_fp_env="${_fp_root}/.conda-envs/foundationpose5070"

export PATH="${_fp_env}/bin:${PATH}"
export CUDA_HOME="${_fp_env}"
export CPATH="${_fp_env}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${_fp_env}/targets/x86_64-linux/lib:${_fp_env}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${_fp_env}/targets/x86_64-linux/lib:${_fp_env}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

unset _fp_root
unset _fp_env
