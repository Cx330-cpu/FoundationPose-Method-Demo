#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --network host \
  -f "${DIR}/dockerfile.5070ti" \
  -t foundationpose:5070ti \
  "${DIR}/.."
