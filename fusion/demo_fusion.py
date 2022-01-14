from SharedDetection import SharedDetection, TreeMemory
from numpy import expand_dims
import cv2
from copy import deepcopy
import numpy as np
import os
import re
from time import sleep, time
from pathlib import Path
from keras.models import load_model
import asyncio
from fusion import new_score2, save_tree
from queue import Queue
from threading import Thread
from copy import deepcopy
import logging

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
    sem_mask_resize = cv2.resize(sem_mask, (dim[1], dim[0]))
    # print(sem_mask_resize.shape, " elaspesd {:.3f}".format(time() - start))
    return sem_mask_resize


class AsyncDetection(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.root = TreeMemory()
        self.terminated = False
        self.count = 0


    def fuse(self, nameId):
        if not fileIdentifier.get(nameId):return
        print("[+fuse] updating tree for %d" % nameId)
        ped_mask = get_pedstrians(fileIdentifier[nameId])
        node = self.root.search(nameId)
        if node and len(node.payload) > 0 and len(node.payload) % node.ncol == 0 and len(node.payload) >= node.ncol:
            value = [node.payload[i:node.ncol + i] for i in range(0, len(node.payload), node.ncol)]
            for i, pred in enumerate(deepcopy(value)):
                if len(pred) != fusedTree.ncol:
                    logging.error("[+ fuse ] payload mismatch %d != %d" %(len(pred), fusedTree.ncol))
                    continue
                score, bb = pred[0], pred[1:]
                try:
                    updateScore = new_score2(ped_mask, bb, score)
                    # update node information
                    # node.payload[i * 5] = updateScore
                    pred[0] = updateScore
                    fusedTree.insert(nameId, node.size, pred)
                    if updateScore > score and updateScore > 0.5 and score <= 0.5:
                        self.count += 1
                        print("[+] %d found improvement on " % self.count, nameId, score, updateScore)
                except:
                    logging.error("[+ fuse] fileID = %d update cannot perform"%nameId)
        else:
            # put back into the queue for later processing
            q.put(nameId)




    def run(self):
        shdr = SharedDetection(buffer_size=7, read_only=True)
        indxCounter = set()
        lastSeen = 0
        os.system('clear')
        print('starting pix2pix')
        for c in shdr.read_shm_memory():
            # print('[+] parsing shared memory ', len(c), c.tolist() )
            payload = c[2:].tolist()
            index, size = int(c[0]), int(c[1])
            if len(payload) == self.root.ncol:
                self.root.insert(index, size, payload)
                indxCounter.add(index)
                if lastSeen != len(indxCounter):
                    q.put(index)
                    lastSeen = len(indxCounter)
                    print("[+] processed items = %d key = %d "% (lastSeen, index))

        # self.root.PrintTree()
        self.terminated = True
        print("[+] demo fusion terminated")



if __name__ == '__main__':
    # config pix2pix gan
    dataset = "/home/redwan/PyDev/Pedestron/data"
    model = PixNdPixGAN("/home/redwan/PyDev/Pix2PixGAN/newmodel.h5")
    fileIdentifier = {}
    for filename in Path(dataset).glob('*.*'):
        nameId = int(re.findall(r'(\d+)', os.path.basename(filename.name))[0])
        fileIdentifier[nameId] = str(filename)

    q = Queue()
    fusedTree = TreeMemory()
    detector = AsyncDetection()
    detector.start()
    while not detector.terminated:
        while not q.empty():
            item = q.get()
            detector.fuse(item)
    detector.join()

    fusedTree.PrintTree()
    print('[+] saving results ...')
    folder = 'results/demo'
    os.makedirs(folder, exist_ok=True)
    save_tree(fusedTree, folder, fileIdentifier)
    print('[+] tree saved')
    print("[+] fused tree terminated")




