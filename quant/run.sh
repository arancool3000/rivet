#!/bin/sh
# usage: run.sh phase1.py [args]   (resumable; kill to pause, re-run to resume)
V=$HOME/.lmstudio/extensions/backends/vendor/_amphibian
cd "$(dirname "$0")" && PYTHONPATH=$V/app-mlx-generate-mac26-arm64@41/lib/python3.11/site-packages exec $V/cpython3.11-mac-arm64@11/bin/python3.11 "$@"
