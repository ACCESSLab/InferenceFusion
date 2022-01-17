#!/usr/bin/env bash

OUTDIR='demoResults' # output result directory
# test image direcotry: supported image extensions are (.jpg or .png)
# make sure images are chronologically sorted  
# numeric names of images are preferable 
DATASET="../datasets/greensboro/image" 

# this function is responsbile for running pedestrian detection branch 
docker_pedestron()
{
    # since we have to run two process in parallel, run the docker container in detach mode 
    # besides dataset here we need to map two custom scripts paths to the docker container
    # /pedestron/data is test data direcotry 
    # /pedestron/tools/demo_eval.py is responsible to transfer the object detector's inference to pix2pixgan using shared memory
    # SharedDetection.py is a custom datastructure for fusing joint detection information 
    # result_demo is the output directory of joint detection 
    # /pedestron/pretrain_model is the object detectors' pretrained weights directory 
    # supported detectors are cascade_hrnet, cascade_mobilenet, csp, mgan
    docker run --gpus '"device=0,1,3"'  --shm-size=8g -d --rm --name pedestron \
        --ipc=host --net=host\
        -v "$(pwd)/detect.sh":/pedestron/detect.sh \
        -v "$(pwd)/$DATASET":/pedestron/data \
        -v $(pwd)/demo_shm.py:/pedestron/tools/demo_eval.py \
        -v $(pwd)/SharedDetection.py:/pedestron/tools/SharedDetection.py \
        -v $OUTDIR:/pedestron/result_demo \
        -v $(pwd)/../detectors/pretrained_models:/pedestron/pretrain_model \
        pedestron:2.1 bash -c "/pedestron/detect.sh $1" 
}

# this function is responsbile for running pedestrian mask segmentation branch and inference fusion  
conda_pix2pix()
{
    # pix2pixGAN use shared memory as a read only mode, need to wait a bit until the object detector branch starts 
    sleep 3
    echo "[+] executing shm_parser"
    # make sure you configure pix2pix gan properly and use the python which is configured for the pix2pix GAN 
    PYTHON="/home/redwan/anaconda3/envs/Pix2PixGAN/bin/python"
    # note that we use GPU id 0, 1, 3 for the object detection branch and GPU id 2 for the semantic segmentation branch
    sudo CUDA_VISIBLE_DEVICES=2 $PYTHON demo_fusion.py \
    --dataset $DATASET\
    --model "../semantics/pretrained_models/checkpoints/pretrained/pix2pix/weight_pix2pix_cityscape.h5"
}

# docker pedestron required object detector
# available options are cascade_hrnet, cascade_mobilenet, csp, mgan
docker_pedestron "cascade_mobilenet"
conda_pix2pix




