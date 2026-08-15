#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/env_5070ti.sh"

cd "${ROOT}"

args=("$@")
if [[ "${args[0]:-}" == "run_demo.py" ]]; then
  has_debug_dir=0
  for arg in "${args[@]}"; do
    if [[ "${arg}" == "--debug_dir" || "${arg}" == --debug_dir=* ]]; then
      has_debug_dir=1
      break
    fi
  done

  if [[ "${has_debug_dir}" == "0" ]]; then
    args+=("--debug_dir" "${ROOT}/debug_5070ti")
  fi
fi

exec "${ROOT}/.conda-envs/foundationpose5070/bin/python" "${args[@]}"
