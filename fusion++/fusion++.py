import cv2
import os
import pickle
import numpy as np

from copy import deepcopy
from pathlib import Path
from collections import Counter
from argparse import ArgumentParser
import asyncio
import multiprocessing

# visualize fusion results with open cv

cyan_color = (255, 255, 0)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)


def overlay_mask(image, mask, color):
    color_mask = np.zeros(image.shape, dtype=np.uint8)
    color_mask[:, :, 0] = color[0]
    color_mask[:, :, 1] = color[1]
    color_mask[:, :, 2] = color[2]
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    image = cv2.addWeighted(image, 1, color_mask, 0.5, 0.5)
    return image


def overlay_box(image, box, color):
    thickness = 2
    bb = list(map(int, box))
    start_point = tuple(bb[:2])
    end_point = tuple(bb[2:])
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


def show_result(fused_data, display=True):
    image = fused_data.image
    for gt in fused_data.gt:
        image = overlay_box(image, gt.box, green_color)

    for ft in fused_data.ft:
        if ft.conf >= 0.5:
            image = overlay_box(image, ft.box, red_color)

    for dt in fused_data.dt:
        if dt.conf >= 0.5:
            image = overlay_box(image, dt.box, blue_color)

    mask = fused_data.mask
    image = overlay_mask(image, mask, cyan_color)
    if display:
        cv2.imshow('frame', image)
        cv2.waitKey(0)
    return image


# core of fusion function starts from here

class Object:

    def __init__(self, txt_file, label=None, box=None, conf=None):
        self.filename = os.path.basename(txt_file)
        self.label = label
        self.conf = conf
        try:
            box = list(map(int, box))
        except:
            box = list(map(float, box))
            box = list(map(int, box))
        self.box = box

    @classmethod
    def parse_file(cls, txt_file):
        with open(txt_file) as file:
            data = file.readlines()
        for line in data:
            line = line.strip()
            words = line.split(" ")

            if len(words) == 5:
                cat, box = words[0], words[1:]
                yield cls(txt_file, cat, box)
            elif len(words) == 6:
                cat, cls_name, box = words[0], words[1], words[2:]
                yield cls(txt_file, cat, box, float(cls_name))


def bb_to_area(bb):
    """
    width : float Rectangle width
    height : float Rectangle height
    :param bb: xmin    ymin    xmax    ymax
    :return:
    """
    width = bb[2] - bb[0]
    height = bb[3] - bb[1]
    area = width * height
    return area


def new_score2(mask, bb, score):
    # detect gan inference based on a bb
    if score <= 0.1:
        return score

    Area = max(bb_to_area(bb), 1e-10)
    box = np.array(bb, dtype=int)
    # box is given in (xmin, ymin, xmax, ymax)
    # mask_bb = mask[box[0]:box[2], box[1]:box[3]] is wrong
    # need to convert in (ymin, ymax, xmin, xmax)
    mask_bb = mask[box[1]:box[3], box[0]:box[2]]
    mask_bb[mask_bb < 255] = 0
    N = np.sum(np.sum(mask_bb / 255.0, axis=1), axis=0)
    M = (Area - N) * score  # other pixels remain default score
    prob = (M + N) / Area  # normalize the score
    return min(prob, 1.0)


class FusionData:

    def __init__(self, image, mask, gt, dt):
        self.image = image
        self.mask = mask
        self.gt = list(gt)
        self.dt = list(dt)
        self.filename = self.gt[0].filename
        self.ft = deepcopy(self.dt)
        self.update()

    def update(self):
        for i, dt in enumerate(self.dt):
            score = new_score2(self.mask, dt.box, dt.conf)
            self.ft[i].conf = score

    def __str__(self):
        data = []
        for ft in self.ft:
            try:
                bb = list(map(int, ft.box))
            except:
                ft.box = list(map(float, ft.box))
                bb = list(map(int, ft.box))
            temp = "{} {:.4f} {} {} {} {}".format("person", ft.conf, bb[0], bb[1], bb[2], bb[3])
            data.append(temp)
        return "\n".join(data)

    def __repr__(self):
        return str(self)

    def save_result(self, out_file):
        with open(out_file, 'w+') as file:
            file.write(str(self))

    def save_image(self, img_path):
        image = show_result(self)
        cv2.imwrite(img_path, image)


# file manager starts from here

def read_pkl_data(semantic_map_file):
    with open(semantic_map_file, 'rb') as handle:
        smap = pickle.load(handle)
    return smap


def infer_extension(image_dir):
    exts = [path.name.split('.')[-1] for path in Path(image_dir).glob('*.*')]
    stat = Counter(exts)
    max_ext, max_val = '', 0
    for ext, val in stat.items():
        if max_val < val:
            max_val = val
            max_ext = ext
    return ".{}".format(max_ext)


class FuseFileManager:
    def __init__(self, inp_detection_file, inp_label_file, inp_img_file, out_txt_file, out_img_file):
        self.inp_detection_file = inp_detection_file
        self.inp_label_file = inp_label_file
        self.inp_img_file = inp_img_file
        self.out_txt_file = out_txt_file
        self.out_img_file = out_img_file

    @classmethod
    def parse_dir(cls, inp_detection_dir, inp_label_dir, inp_img_dir, out_txt_dir, out_img_dir):
        ext = infer_extension(inp_img_dir)

        assert ext == '.jpg' or ext == '.png'

        for file in Path(inp_detection_dir).glob('*.txt'):
            basefile = file.name
            imgfile = basefile.replace('.txt', ext)
            inp_txt_file = os.path.join(inp_label_dir, basefile)
            inp_img_file = os.path.join(inp_img_dir, imgfile)

            out_txt_file = os.path.join(out_txt_dir, basefile)
            out_img_file = os.path.join(out_img_dir, imgfile)

            yield cls(str(file), inp_txt_file, inp_img_file, out_txt_file, out_img_file)

    def __str__(self):
        return f"[+] out_txt_file = {self.out_txt_file} | out_img_file = {self.out_img_file}"

    def __repr__(self):
        return str(self)


async def execute_fuse_file(file):
    global semantic_map

    # load image
    image = cv2.imread(file.inp_img_file)
    img_name = os.path.basename(file.inp_img_file)

    # load and resize corresponding mask
    if img_name not in semantic_map:
        img_name = [int(s) for s in img_name.split('.') if s.isdigit()][0]

    mask = semantic_map[img_name]['mask']
    dim = image.shape
    mask = cv2.resize(mask, (dim[1], dim[0]))

    # combine fusion result for each file
    gt_labels = Object.parse_file(file.inp_label_file)
    dt_labels = Object.parse_file(file.inp_detection_file)
    fused_data = FusionData(image, mask, gt_labels, dt_labels)

    # save result
    fused_data.save_result(file.out_txt_file)

    # save image
    if args.save_img:
        fused_data.save_image(file.out_img_file)

    await asyncio.sleep(0.0001)
    print('[+] fusion result computed', fused_data.filename)


class TaskManager(multiprocessing.Process):
    # https://medium.com/@nbasker/python-asyncio-with-multiprocessing-2595f8ee3f8
    def __init__(self, file):
        multiprocessing.Process.__init__(self)
        self.file = file

    def run(self) -> None:
        asyncio.run(execute_fuse_file(self.file))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pkl', type=str, default='data/GBD/sem_map_seg_fromer_greensboro.pkl')
    parser.add_argument('--detection_dir', type=str, default='data/GBD/ssd_mobilenet_fpn/')
    parser.add_argument('--label_dir', type=str, default='data/GBD/label/')
    parser.add_argument('--img_dir', type=str, default='data/GBD/img/')
    parser.add_argument('--out_dir', type=str, default='data/GBD/ssd_mobilenet_fpn/')
    parser.add_argument('--save_img', action='store_true', help='will save image if this argument is provided')

    args = parser.parse_args()

    semantic_map = read_pkl_data(args.pkl)

    out_txt_dir = os.path.join(args.out_dir, 'fusion_results')
    os.makedirs(out_txt_dir, exist_ok=True)

    out_img_dir = os.path.join(args.out_dir, 'fusion_images')
    if args.save_img: os.makedirs(out_img_dir, exist_ok=True)

    files = FuseFileManager.parse_dir(args.detection_dir, args.label_dir, args.img_dir, out_txt_dir, out_img_dir)

    tasks = (TaskManager(f) for f in files)

    for t in tasks:
        t.start()

    print('[+] fusion terminated ... ')
