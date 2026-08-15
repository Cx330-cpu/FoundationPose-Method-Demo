#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEBUG_DIR="${DEBUG_DIR:-debug_5070ti_docker}"
export FOUNDATIONPOSE_MYCPP_BUILD_DIR="${FOUNDATIONPOSE_MYCPP_BUILD_DIR:-${ROOT}/mycpp/build_5070ti_docker}"

python check_env.py --demo-data
python run_demo.py --debug "${DEBUG:-1}" --debug_dir "${DEBUG_DIR}" "$@"
