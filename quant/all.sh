#!/bin/sh
# Full run, resumable: each step is skipped if its result is already in progress.log
cd "$(dirname "$0")"
P=$HOME/.lmstudio/models/aran
done_() { grep -q "$1" progress.log; }
done_ "QUIZ pi5:" || ./run.sh quiz.py $P/1-Apex-Gemma4-26B pi5
done_ "QUIZ pi4:" || ./run.sh quiz.py $P/4-Pi4-Gemma4-26B-3bit pi4
for c in "B 3*12,2*18" "C 2*30" "A 3*30"; do
  set -- $c
  done_ "$1 TUNE DONE" || ./run.sh tune.py $1 "$2" 150 || exit 1
  done_ "QUIZ $1:" || ./run.sh quiz.py out/$1 $1
done
echo "$(date +%H:%M:%S) ALL DONE" >> progress.log
