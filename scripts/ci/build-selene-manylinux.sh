#!/usr/bin/env bash
# Run inside the release workflow's pinned manylinux_2_28 image.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bash scripts/ci/ensure-rust.sh stable minimal
export PATH="$HOME/.cargo/bin:/opt/python/cp312-cp312/bin:$PATH"
/opt/python/cp312-cp312/bin/python -m pip install build
/opt/python/cp312-cp312/bin/python scripts/ci/release_selene_wheels.py build
