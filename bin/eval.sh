#!/usr/bin/zsh

#time python eval.py -na -np --gt gt --dr dt
ROOT="$(pwd)"
gt="total_gt"
dt="total_dt"
# time ../cmake-build-debug/mAP --gt gt --dt dt --iou 0.5
python eval.py -na -np --gt $ROOT/$gt --dr $ROOT/$dt --set-class-iou person 0.5
