#!/usr/bin/env bash
OUTDIR="$(pwd)/../results/CityPersons"
data="/home/redwan/PyDev/InferenceFusion/groundTruth/CityPersons/val_img"
docker_pedestron()
{
    TEMP="$OUTDIR/$1"
    mkdir -p TEMP
    docker run --gpus '"device=0,1,3"'  --shm-size=8g -it --rm --name pedestron \
        --ipc=host --net=host\
        -v $(pwd)/detect.sh:/pedestron/detect.sh \
        -v $data:/pedestron/data \
        -v $(pwd)/demo_eval.py:/pedestron/tools/demo_eval.py \
        -v TEMP:/pedestron/result_demo \
        -v $(pwd)/../detectors/pretrained_models:/pedestron/pretrain_model \
        pedestron:2.1 bash -c "/pedestron/detect.sh $1"
}

#docker_pedestron

for det in 'cascade_hrnet' 'cascade_mobilenet' 'csp' 'mgan'
do
  echo "[+] running detector $det"
  docker_pedestron $det
done

