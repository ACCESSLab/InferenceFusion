#!/usr/bin/env bash 

exe="./mAP"
ROOT="$(pwd)/.."

eval()
{
  gt="$ROOT/results/$1/$1_gt"
  dt="$ROOT/results/$1/$1_dt"
  $exe --gt $gt --dt $dt --iou 0.5
}

#eval "F-DNN"
#eval "F-DNN-SS"
#eval "F-DNN2-SS"

eval "SDS-RCNN"