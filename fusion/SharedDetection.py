'''
    This library will enable sharing pedestrian detection results to another
    processor (e.g., semantic segmentation network). Pedestrian detection is
    implemented using docker container pedestron (https://github.com/hasanirtiza/Pedestron).
    This library bridge between pedestrian detection and semantic segmentation network.
    Please refer to tools/demo_shm.py for a detailed implementation.
    REQUIREMENTS:
        - SharedArray==3.2.1
        - numpy
    LIBS:
        - TreeMemory:
            used for storing detection result. A detection result comprsies with following information
            msg = [fileIdentifier, size, confScore, BoundingBox]
            index -> fileIndentifier - string name of the image file or index of the image.
                              we use regex to extract integer index from a string filename
            size -> #objects - number of detected objects in an input image. An image might contain multiple pedestrians but
                   we cannot share all the detection results at the same time. Therefore, this class will structure,
                   group and organige the detected individual pedestrian detectection results based on their fileIndentifier
            payload -> [confScore, BoundingBox]
                confScore - confidence score for each bounding box
                BoundingBox - bounding box for a single pedestrian comprises with 4 elements.
                            First two elements represent bottom right corner point
                            last two elements represent top left corner point
                            This definition is same for cv2 rectangle function
        - SharedDetection:
            this class is responsible for exchanging information from docker container to the target
            application. The key idea is to cache numpy array and trigger an event when change occurs.\
            SharedDetection is implemented using SharedArray. We need to make sure

'''

from copy import deepcopy
import asyncio
from pprint import pprint
import logging

class TreeMemory:
    ncol = 5
    def __init__(self, index=None, size=None, payload=[]):
        '''
        :param index: filename (integer index for filename)
        each file represent an image
        :param size: number of objects in the image
        :param payload:
        detection results. Each image might have multiple objects
        Each object is represented by [confscore, boundingBox]
        TreeMemory class will sort the data based on the filename
        This is an extention of binary tree search
        '''
        self.left = None
        self.right = None
        self.index = index
        self.payload = payload
        self.size = size

    def insert(self, index, size, payload):
        assert isinstance(payload, list)
        assert isinstance(index, int)

        if self.index:
            if index == self.index:
                if len(payload) == self.ncol:
                    self.payload += payload
                else:
                    logging.error("[+TreeMemrory]  payload %d size does not match with ncol"%len(payload))
            elif index < self.index:
                if self.left is None:
                    self.left = TreeMemory(index, size, payload)
                else:
                    self.left.insert(index, size, payload)
            elif index > self.index:
                if self.right is None:
                    self.right = TreeMemory(index, size, payload)
                else:
                    self.right.insert(index, size, payload)
        else:
            self.index = index
            self.payload = payload
            self.size = size

    def PrintTree(self):
        if not self.size:
            return
        if self.left:
            self.left.PrintTree()
        # if self and len(self.payload) > 0 and len(self.payload) % self.ncol == 0 and len(self.payload) >= self.ncol:
        print("[+ Node {}]: size = {}".format(self.index, self.size))
        # pprint(value)
        if self.right:
            self.right.PrintTree()

    def search(self, index):
        if self and index == self.index:
            return self
        if index < self.index:
            if self.left:
                return self.left.search(index)
        else:
            if self.right:
                return self.right.search(index)

    def isFull(self, index):
        node = self.search(index)
        if node.size is None:
            return False
        value, size = node.payload, node.size
        payloadSize = len(value) // self.ncol
        return size == payloadSize

    def getKeys(self):
        keys = []

        def inorder_traverse(root):
            if not root:
                return
            inorder_traverse(root.left)
            keys.append(root.index)
            inorder_traverse(root.right)

        inorder_traverse(deepcopy(self))
        return keys


import numpy as np
import SharedArray as sa
import time


class SharedDetection:
    def __init__(self, buffer_size, read_only=False):
        self.buffer_size = buffer_size
        self.read_only = read_only
        if not self.read_only:
            try:
                sa.delete("sharedDetection")
            except:
                pass
            self.buffer = sa.create("shm://sharedDetection", self.buffer_size)
            time.sleep(1)
        else:
            self.buffer = sa.attach("shm://sharedDetection")

    def __call__(self, value):
        self.buffer[:] = value[:]
        time.sleep(0.0001)

    def stop(self):
        value = np.arange(self.buffer_size, dtype=np.float16)
        value[:] = -1
        self(value)
        time.sleep(0.5)
        sa.delete("sharedDetection")

    def read_shm_memory(self):
        assert self.read_only, 'class needs to be initiated with read_only option'
        cache = []
        while True:
            try:
                c = self.buffer
                if c.tolist() not in cache:
                    if len(cache):
                        cache.pop()
                    if sum(c) == -1 * self.buffer_size:
                        break
                    yield c
                    cache.append(c.tolist())
                # del c
            except:
                break
        print("[+] sharedDetection terminated")



