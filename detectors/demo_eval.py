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
    return ms_bbox_result


def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def run_detector_on_dataset():
    args = parse_args()
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(input_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.png'))
    print(eval_imgs)

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    # prog_bar = mmcv.ProgressBar(len(eval_imgs))
    N = len(eval_imgs)
    for i, im in enumerate(eval_imgs):
        detections = mock_detector(model, im, output_dir)
        current_path = im.replace('.png', '.txt')
        out_file = os.path.basename(current_path)
        out_file = os.path.join(args.output_dir, out_file)

        all_objects = []
        nameId = int(re.findall(r'(\d+)', os.path.basename(im))[0])
        for objects in detections:
            pedestrians = list(filter(lambda x: x[-1] > 0.0, objects))
            # person 0.471781 0 13 174 244
            for obj in pedestrians:
                conf, bb = obj[-1], obj[:-1]
                bb = list(map(str, map(int, bb)))  # list of string
                item = ['person', str(conf)] + bb
                item = " ".join(item)
                all_objects.append(item)

        all_objects = "\n".join(all_objects)
        with open(out_file, 'w+') as file:
            file.write(all_objects)
        # prog_bar.update()
        print("{}/{}".format(i, N))


if __name__ == '__main__':
    run_detector_on_dataset()
    print("[+] pedestron terminated")
    print("")





