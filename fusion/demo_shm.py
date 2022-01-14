import argparse
import re
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import time
import cv2
import torch
import glob
import json
import mmcv
from copy import deepcopy
from mmdet.apis import inference_detector, init_detector, show_result
from SharedDetection import SharedDetection

BUFFER_SIZE = 7


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_img_dir', type=str, help='the dir of input images')
    parser.add_argument('output_dir', type=str, help='the dir for result images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    args = parser.parse_args()
    return args


def mock_detector(model, image_name, output_dir):
    image = cv2.imread(image_name)
    results = inference_detector(model, image)
    if len(results) == 2:
        ms_bbox_result, ms_segm_result = deepcopy(results)
    else:
        ms_bbox_result = deepcopy(results)

    # basename = os.path.basename(image_name).split('.')[0]
    # result_name = basename + "_result.jpg"
    # result_name = os.path.join(output_dir, result_name)
    # show_result(image, results, model.CLASSES, out_file=result_name)

    return ms_bbox_result


def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def share_msg(nameId, objects):
    objects = objects.tolist()
    pedestrians = list(filter(lambda x: x[-1] > 0.1, objects))
    size = len(pedestrians)
    for obj in pedestrians:
        value = [nameId, size, obj[-1]] + obj[:-1]
        sd(value)


def run_detector_on_dataset():
    args = parse_args()
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print(input_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.png'))
    # print(eval_imgs)
    print("[+] total images = %d at %s " %(len(eval_imgs), input_dir))

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    prog_bar = mmcv.ProgressBar(len(eval_imgs))
    for im in eval_imgs:
        detections = mock_detector(model, im, output_dir)
        nameId = int(re.findall(r'(\d+)', os.path.basename(im))[0])
        for objects in detections:
            share_msg(nameId, objects)
        prog_bar.update()


if __name__ == '__main__':
    sd = SharedDetection(buffer_size=BUFFER_SIZE)
    run_detector_on_dataset()
    sd.stop()
    print("[+] pedestron terminated")





