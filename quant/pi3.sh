#!/bin/sh
cd "$(dirname "$0")"
grep -q "QUIZ pi3:" progress.log || ./run.sh quiz.py $HOME/.lmstudio/models/aran/2-Forge-gpt-oss-20b pi3
grep -q "P3 TUNE DONE" progress.log || ./run.sh pi3.py 75 || { echo "$(date +%H:%M:%S) P3 FAILED" >> progress.log; exit 1; }
grep -q "QUIZ P3:" progress.log || ./run.sh quiz.py out/P3 P3
echo "$(date +%H:%M:%S) P3 ALL DONE" >> progress.log
