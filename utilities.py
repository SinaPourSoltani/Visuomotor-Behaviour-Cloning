from PIL import Image
from dataclasses import dataclass
import numpy as np

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
    def __init__(self, steps):
        self.steps = steps
        self.pokes = np.empty((steps, 2))

    def add(self, image: Image, poke: np.ndarray, idx):
        # TODO: handle dynamic filepath given as argument
        image.save('data/images/img' + str(idx).zfill(4) + '.png')
        self.pokes[idx] = poke

    def save_pokes(self):
        # TODO: handle dynamic filepath given as argument
        np.save('data/pokes.npy', self.pokes)



