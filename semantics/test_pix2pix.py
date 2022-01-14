#!/home/redwan/anaconda3/envs/Pix2PixGAN/bin/python

import pickle
from numpy import expand_dims
import cv2
from copy import deepcopy
import numpy as np
import os
import re
from time import sleep, time
from pathlib import Path
from keras.models import load_model

from argparse import ArgumentParser

np.random.seed(2020198875)


class PixNdPixGAN(object):
    def __init__(self, gen_weight_path):
        # model_ = load_model(gen_weight_path,custom_objects={'define_encoder_block': define_encoder_block}, compile=False)

        self.model = load_model(gen_weight_path)

    def __call__(self, *args, **kwargs):
        img = deepcopy(np.squeeze(args[0]))
        src_image = cv2.resize(img, (256, 256))
        img = deepcopy(src_image)
        src_image = (src_image - 127.5) / 127.5
        src_image = expand_dims(src_image, 0)
        gen_image = self.model.predict(src_image)
        gen_image = ((gen_image + 1) / 2.0) * 255
        gen_image = np.squeeze(gen_image)
        return gen_image

    def get_mask(self, gan_frame):
        lower_red = np.array([158, 185, 141])
        upper_red = np.array([255, 255, 255])
        gan_frame = np.array(gan_frame, dtype=np.uint8)
        hsv = cv2.cvtColor(gan_frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return mask


def get_pedstrians(filename):
    assert isinstance(filename, str)
    img = cv2.imread(filename)
    dim = img.shape
    start = time()
    sem_map = model(img)
    sem_mask = model.get_mask(sem_map)

    print("[+ pix2pix]", indx, sem_mask.shape, " elaspesd {:.3f}".format(time() - start))
    return sem_mask, dim


if __name__ == "__main__":
    print("[+] pix2pix gan running")
    dataset = "/home/mashuk2/Desktop/ITS_benchmark/InferenceFusion/fusion/groundTruth/CityPersons/val_img/"
    model = PixNdPixGAN("pretrained_models/weight_pix2pix_cityscape.h5")
    results = {}
    for indx, filename in enumerate(Path(dataset).glob('*.*')):
        nameId = int(re.findall(r'(\d+)', os.path.basename(filename.name))[0])
        # nameId = hash(filename.name.split('.')[0])
        ped_mask, imgDim = get_pedstrians(str(filename))
        results[nameId] = {"mask":ped_mask, "ori_dim":imgDim}

    with open('sem_map_pix2pix_city.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
