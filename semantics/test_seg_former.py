import torch
import random
import gdown
import cv2
import argparse
import pickle
import os
import glob
import re
import numpy as np
from torchvision import io
from torchvision import transforms as T
from PIL import Image
from utils.utils import show_models
from pathlib import Path
from models import get_model
from datasets import ADE20K
from models import backbones
from time import sleep, time
from collections import namedtuple

def predict_semantic_map(img, model):
    '''

    :param img: Input RGB image
    :param model: semantic segmentation model
    :return: semantic segmentation image
    '''
    image = T.Resize((512, 512))(img)
    # scale to [0.0, 1.0]
    image = image.float() / 255
    # normalize
    image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
    # add batch size
    image = image.unsqueeze(0)

    with torch.no_grad():
        seg = model(image)

    seg = seg.softmax(1).argmax(1).to(int)
    seg.unique()
    palette = ADE20K.PALETTE
    seg_map = palette[seg].squeeze().to(torch.uint8)

    np_image = seg_map.cpu().detach().numpy()
    
    return np_image

def pedestrian_mask(semantic_frame):
    '''

    :param semantic_frame: semantic segmentation image
    :return: mask of the pedestrian in grayscale image
    '''
    lower_purple = np.array([158, 185, 141])
    upper_purple = np.array([255, 255, 255])
    semantic_frame = np.array(semantic_frame, dtype=np.uint8)
    hsv = cv2.cvtColor(semantic_frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    return mask

data = namedtuple('data', 'filename file_id dim image')
def load_dataset(input_data_path):
    '''

    :param input_data_path: input image directory containing jpg or png images
    :return:  instance of data which is a generator
    '''
    for ext in ['jpg', 'png']:
        for indx, filename in enumerate(Path(input_data_path).glob('*.%s'%ext)):
            nameId = int(re.findall(r'(\d+)', os.path.basename(filename.name))[0])
            # img = cv2.imread(str(filename))
            img = io.read_image(str(filename))
            c, H, W  = img.shape
            image_data = data(filename=filename.name, file_id=nameId, dim=[H, W, c], image=img)
            yield image_data
          
def load_model(weight):
    model = get_model(
        model_name='SegFormer',
        variant='B3',
        num_classes=150  # ade20k
    )
    try:
        model.load_state_dict(torch.load(weight, map_location='cpu'))
    except:
        print("Download a pretrained model's weights from the result table.")
        exit(-1)
    model.eval()
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--images", type=str, default='/home/mashuk2/Desktop/ITS_benchmark/InferenceFusion/fusion/groundTruth/CityPersons/val_img/')
    parser.add_argument("--weight", type=str, default='checkpoints/pretrained/segformer/segformer.b3.ade.pth')
    parser.add_argument("--output", type=str, default='./all_pkl_files/sem_map_seg_fromer_greensboro2.pkl')

    args = parser.parse_args()
    print(args.images)
    input_data_path = args.images
    model = load_model(args.weight)
    results = {}
    count = 0
    for file in load_dataset(input_data_path):
        semantic_frame = predict_semantic_map(file.image, model)
       
        ped_mask = pedestrian_mask(semantic_frame.copy())

        dim = (file.dim[1],file.dim[0])
   
        results[file.filename] = {"mask":ped_mask, "ori_dim":dim, "s_map":semantic_frame}

         
        print(file.filename, file.dim)
         
    with open(args.output, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    


