from PIL import Image
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import os
from os import path

@dataclass
class Item:
    pos: np.ndarray
    ori: np.ndarray
    dim: np.ndarray

    def set_dim(self, dim):
        self.dim = np.asarray(dim)

    def set_pos(self, pos):
        self.pos = np.asarray(pos)

    def set_ori(self, ori):
        self.ori = np.asarray(ori)

    def set_pos_and_ori_from_pose(self, pose):
        self.set_pos(pose[0])
        self.set_ori(pose[1])

@dataclass
class State:
    image: Image
    item: Item
    goal: Item


class Dataset:
    def __init__(self, file_name, image_path=None, filemode="x"):
        
        self.idx = 0
        self.dataID = datetime.now().strftime("%d-%m_%H:%M:%S")
        self.path_to_store_img = image_path

        if image_path == None: 
            self.path_to_store_img = "data/images/" + self.dataID + "/"
        else: 
            self.path_to_store_img = image_path


        if file_name[-4:] == ".npy":
            self.file_name = file_name
        else: 
            self.file_name = file_name + ".npy"

        try: 
            self.file = open(self.file_name, filemode)
        except: 
            raise Exception("Datafile was not created")
        
        try:
            os.makedirs(self.path_to_store_img)
        except:
            print("Directory already exists")
        

    def __del__(self):
        self.file.close()

    def add(self, image: Image, poke: np.ndarray, idx):
        image_name = self.dataID + "_" + str(self.idx).zfill(4) + '.png'
        image.save(self.path_to_store_img + image_name)
        self.file.write(image_name + ", " + str(poke[0]) + ", " + str(poke[1]) + "\n") 
        self.idx += 1


class Geometry:
    @staticmethod
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def get_direction_vector(from_pos, to_pos):
        dir_vec = to_pos - from_pos
        unit_dir = dir_vec / np.linalg.norm(dir_vec)
        return unit_dir

    @staticmethod
    def rotate_vector(v, a):
        return np.asarray([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])

    @staticmethod
    def angle_between_vectors(v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


