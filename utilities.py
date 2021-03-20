from PIL import Image
from dataclasses import dataclass

@dataclass
class Item:
    pos: list
    ori: list
    dim: list

    def set_dim(self, dim):
        self.dim = list(dim)

    def set_pos(self, pos):
        self.pos = list(pos)

    def set_ori(self, ori):
        self.ori = list(ori)

    def set_pos_and_ori_from_pose(self, pose):
        self.set_pos(pose[0])
        self.set_ori(pose[1])

@dataclass
class State:
    image: Image
    item: Item
    goal: Item
