#!/usr/bin/env bash

#!/usr/bin/env bash
OUTDIR="../results/CascadeM-RCNN-Caltech"
data="/home/redwan/PycharmProjects/caltechBenchmark/results/F-DNN/test_images"
ROOT="/home/redwan/Desktop/MashukBhai/InferenceFusion/detectors"
mkdir $OUTDIR

docker_pedestron()
{
    TEMP="$OUTDIR/$1"
    mkdir -p TEMP
    docker run -it --rm --name pedestron \
        --ipc=host --net=host\
        -v $ROOT/detect.sh:/pedestron/detect.sh \
        -v $data:/pedestron/data \
        -v $ROOT/demo_eval.py:/pedestron/tools/demo_eval.py \
        -v TEMP:/pedestron/result_demo \
        -v $ROOT/../detectors/pretrained_models:/pedestron/pretrain_model \
        pedestron:2.1 bash -c "/pedestron/detect.sh $1"
}

#docker_pedestron

for det in 'caltech_mrcnn'
do
  echo "[+] running detector $det"
  docker_pedestron $det
done
