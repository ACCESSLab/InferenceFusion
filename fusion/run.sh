#!/usr/bin/env bash

OUTDIR='demoResults'
docker_pedestron()
{
    docker run --gpus '"device=0,1,3"'  --shm-size=8g -d --rm --name pedestron \
        --ipc=host --net=host\
        -v $(pwd)/../detectors/detect.sh:/pedestron/detect.sh \
        -v $(pwd)/../data:/pedestron/data \
        -v $(pwd)/demo_shm.py:/pedestron/tools/demo_eval.py \
        -v $(pwd)/SharedDetection.py:/pedestron/tools/SharedDetection.py \
        -v $OUTDIR:/pedestron/result_demo \
        -v $(pwd)/../detectors/pretrained_models:/pedestron/pretrain_model \
        pedestron:2.1 sh /pedestron/detect.sh
}


conda_pix2pix()
{
    sleep 3
    echo "[+] executing shm_parser"
    sudo CUDA_VISIBLE_DEVICES=2 /home/redwan/anaconda3/envs/Pix2PixGAN/bin/python demo_fusion.py
}


coproc(docker_pedestron)
conda_pix2pix
#arg_parser
#echo "[+] removing docker container"
#docker stop pedestron && docker rm pedestron



