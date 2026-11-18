#!/usr/bin/env bash
set -euo pipefail
'/opt/conda/envs/vggts/bin/python' -m pip install --no-deps --no-build-isolation -e '/root/VGGT-S-main'
'/opt/conda/envs/vggts/bin/python' -m pip install --no-deps --no-build-isolation -e '/root/VGGT-S-main/third_party/vggt_main'
