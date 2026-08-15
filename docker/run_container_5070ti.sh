#!/usr/bin/env bash
set -euo pipefail

docker rm -f foundationpose-5070ti >/dev/null 2>&1 || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

docker run --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it \
  -p 5000:5000 \
  --name foundationpose-5070ti \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "${PROJECT_DIR}:${PROJECT_DIR}" \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  --ipc=host \
  -e DISPLAY="${DISPLAY:-}" \
  -e GIT_INDEX_FILE \
  -e TORCH_CUDA_ARCH_LIST=12.0 \
  -e FORCE_CUDA=1 \
  -e PYTHONNOUSERSITE=1 \
  foundationpose:5070ti \
  bash -lc "cd '${PROJECT_DIR}' && bash"
