#!/usr/bin/env bash
cascade_hrnet()
{
  python tools/demo_eval.py configs/elephant/cityperson/cascade_hrnet.py \
    pretrain_model/epoch_5.pth.stu data/ result_demo/
}

cascade_mobilenet()
{
  python tools/demo_eval.py configs/elephant/cityperson/cascade_mobilenet.py \
    pretrain_model/epoch_16.pth.stu data/ result_demo/
}


csp()
{
  python tools/demo_eval.py configs/elephant/cityperson/csp_r50.py \
    pretrain_model/epoch_72.pth.stu data/ result_demo/
}

mgan()
{
  python tools/demo_eval.py configs/elephant/cityperson/mgan_vgg.py \
    pretrain_model/epoch_1.pth data/ result_demo/
}

#cascade_hrnet
#cascade_mobilenet
#csp
#mgan
if [[ -z $1 ]]; then
  echo "[!] needs arguments"
  exit
fi

if [[ $1 == *"cascade_hrnet"* ]]; then
    echo "[+] running cascade_hrnet"
    cascade_hrnet
  elif [[ $1 == *"cascade_mobilenet"* ]]; then
    echo "[+] running cascade_mobilenet"
    cascade_mobilenet
  elif [[ $1 == *"csp"* ]]; then
    echo "[+] running csp"
    csp
  elif [[ $1 == *"mgan"* ]]; then
    echo "[+] running mgan"
    mgan
fi
