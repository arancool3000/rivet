#!/bin/sh
# Overnight: wait until 01:00, then re-save B and C from their checkpoints and run the missing quizzes.
cd "$(dirname "$0")"; P=$HOME/.lmstudio/models/aran
echo "$(date +%H:%M:%S) NIGHT START" >> progress.log
for c in "B 3*12,2*18" "C 2*30"; do
  set -- $c
  ./run.sh tune.py $1 "$2" 150 && ./run.sh quiz.py out/$1 $1
done
./run.sh quiz.py $P/1-Apex-Gemma4-26B pi5
./run.sh quiz.py $P/4-Pi4-Gemma4-26B-3bit pi4
echo "$(date +%H:%M:%S) NIGHT DONE" >> progress.log
