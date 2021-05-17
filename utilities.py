from PIL import Image
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import os
import csv
import shutil
from os import path

def combine_csv(source_csv, dest_csv):
    source_data_file = open(source_csv, 'r')
    reader = csv.reader(source_data_file)
    header = next(reader)
    data = {}

    with open(dest_csv, 'a') as dst_file:
        with open(source_csv, 'r') as src_file:
            src_file.__next__()
            for line in src_file.readlines():
                dst_file.write(line)

def move_to_folder(source,dest,copy=True):
    sub_folders = next(os.walk(source))[1]

    for sf in sub_folders:
        source_sf_path = os.path.join(source,sf)
        source_sf_folder = os.listdir(source_sf_path)

        dest_sf_path = os.path.join(dest,sf)
        os.mkdir(dest_sf_path)


        dest_sf_folder = source_sf_folder

        for file in dest_sf_folder:
            file_path = os.path.join(source_sf_path,file)
            if copy:
                shutil.copy2(file_path,dest_sf_path)
            else:
                shutil.move(file_path,dest_sf_path)

#https://note.nkmk.me/en/python-pillow-concat-images/
def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

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
    image: [Image]
    item: Item
    goal: Item


class Dataset:
    def __init__(self, verbose, stereo_images, file_name, image_path=None, data_file_path=None, filemode="x", start_idx=0):

        self.idx = 0
        self.dataID = datetime.now().strftime("%d-%m_%H:%M:%S")
        self.path_to_store_img = image_path
        self.verbose = verbose
        self.stereo_images = stereo_images

        self.episodeNum = start_idx
        self.data_file_path = data_file_path

        if image_path == None:
            self.path_to_store_img = "data/images/" + self.dataID + "/"
        else:
            self.path_to_store_img = image_path



        if file_name[-4:] == ".csv":
            self.file_name = file_name
        else:
            self.file_name = file_name + ".csv"

        try:
            self.file = open(self.file_name, filemode)
        except:
            raise Exception("Datafile was not created")

        try:
            os.makedirs(self.path_to_store_img)
        except:
            print("Directory already exists")

        try:
            self.file = open(self.data_file_path + self.file_name, filemode)
        except:
            raise Exception("Datafile was not created")
        if start_idx == 0: # if empty add header, else just append
            if not self.stereo_images:
                self.file.write("image_file_name,∆x,∆y,∆z,episode\n")
            else:
                self.file.write("image_file_name_left,image_file_name_right,∆x,∆y,∆z,episode\n")


    def __del__(self):
        self.file.close()

    def next_episode(self):
        self.episodeNum += 1
        self.idx = 0

    def save_image(self, image: Image, tag=None):
        stereo_tag = "_" + tag if tag is not None else ""
        image_name = "img_ep_" + str(self.episodeNum) +"_" + str(self.idx).zfill(4) + stereo_tag + "_" + self.dataID + '.png'
        image.save(self.path_to_store_img + image_name)
        if self.verbose: print("Image file name: ", image_name)
        return image_name

    def add(self, images: [Image], poke: np.ndarray):
        image_name = ""
        if self.stereo_images: # Stereo
            tag = ['l','r']
            for i, img in enumerate(images):
                image_name += self.save_image(img, tag[i])
                if i < 1: image_name += ","
        else:
            image_name = self.save_image(images[0])

        self.file.write(image_name + "," + str(poke[0]) + "," + str(poke[1]) + "," + str(poke[2]) + ","+ str(self.episodeNum) + "\n")
        self.idx += 1


class Geometry:
    @staticmethod
    def dist(p1, p2):
        p1_np = np.asarray(p1)
        p2_np = np.asarray(p2)
        return np.linalg.norm(p1_np - p2_np)

    @staticmethod
    def get_direction_vector(from_pos, to_pos):
        dir_vec = to_pos - from_pos
        unit_dir = dir_vec / np.linalg.norm(dir_vec)
        return unit_dir

    @staticmethod
    def rotate_vector(v, a):
        return np.asarray([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])

    @staticmethod
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def angle_between_vectors(v1, v2):
        eps = 0.0000000001
        norm_prod = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm_prod < eps:
            norm_prod = eps

        return np.arccos(np.min([np.max([np.dot(v1, v2) / norm_prod, -1]), 1]))


    @staticmethod
    def distance_to_line(point, line_point1, line_point2):
        tcp_copy = np.copy(point)
        tcp_copy.resize(3)
        linePoint1_copy = np.copy(line_point1)
        linePoint1_copy.resize(3)
        linePoint2_copy = np.copy(line_point2)
        linePoint2_copy.resize(3)

        d = np.linalg.norm(np.cross(linePoint2_copy - linePoint1_copy, linePoint1_copy - tcp_copy))/np.linalg.norm(linePoint2_copy - linePoint1_copy)
        return d
