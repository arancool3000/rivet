#!/bin/bash
# Usage: ./models/join.sh <model-folder>   (run from the folder holding the downloaded .part files)
set -euo pipefail
M="$1"; HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.lmstudio/models/aran/$M"
mkdir -p "$DEST"
cp "$HERE/$M/"* "$DEST/"
for first in "$M"__*.part00; do
  f="${first#"$M"__}"; f="${f%.part00}"
  echo "joining $f"
  cat "$M"__"$f".part* > "$DEST/$f"
done
cd "$DEST" && shasum -a 256 -c "$HERE/$M/SHA256SUMS"
echo "Done: $DEST"
