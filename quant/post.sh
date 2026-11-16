#!/bin/sh
cd "$(dirname "$0")"; P=$HOME/.lmstudio/models/aran
while pgrep -f all.sh >/dev/null; do sleep 30; done
# B was first saved with the original weights by a bug: re-save it from its checkpoint and quiz it again
sed -i "" "s/QUIZ B:/QUIZ pi5(via B bug):/" progress.log
./run.sh tune.py B "3*12,2*18" 150 && ./run.sh quiz.py out/B B
grep -q "^.* QUIZ pi5" progress.log || ./run.sh quiz.py $P/1-Apex-Gemma4-26B pi5
grep -q "^.* QUIZ pi4:" progress.log || ./run.sh quiz.py $P/4-Pi4-Gemma4-26B-3bit pi4
echo "$(date +%H:%M:%S) POST DONE" >> progress.log
