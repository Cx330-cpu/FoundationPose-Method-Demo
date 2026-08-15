#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT}/mycpp"
rm -rf build_5070ti_docker
mkdir -p build_5070ti_docker
cd build_5070ti_docker
cmake .. \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DPYBIND11_PYTHON_EXECUTABLE="$(command -v python)"
ninja -j"$(nproc)"
