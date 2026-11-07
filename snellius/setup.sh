#!/bin/bash
# One-time setup on a Snellius login node: venv + pinned dependencies.
# From the repo root:   snellius/setup.sh
#
# Facts encoded here so nobody rediscovers them:
#  * python >= 3.13 comes from the 2025 module stack; the bare login shell has
#    no python3.13 and module names need their full toolchain suffix.
#  * torch is installed from the CPU wheel index (no GPUs on thin nodes; the
#    default pip torch drags in CUDA).
#  * The dynaplex wheel from PyPI is self-contained — no modules needed at
#    runtime, so job scripts only have to activate the venv.
set -e
module load 2025 Python/3.13.1-GCCcore-14.2.0
cd "$(dirname "$0")/.."
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
mkdir -p results
echo "ready — submit from the repo root, e.g.:"
echo "    sbatch snellius/train.job dcl --n 2000 --m 500 --h 100"
