from utilities import *
import numpy as np
import pybullet as p


class Expert:
    def __init__(self):
        self.item = None
        self.goal = None
        # pass state of robot -> pos of tcp
        #

    def update_item_and_goal(self, item: Item, goal: Item):
        self.item = item
        self.goal = goal

    def get_direction_vector(self):
        dir_vec = self.goal.pos - self.item.pos
        unit_dir = dir_vec / np.linalg.norm(dir_vec)
        return unit_dir

    def rotate_vector(self, v, a):
        return [np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]]

    def angle_between_vectors(self, v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def calculate_poke2(self, item: Item, goal: Item):
        # line between centers of gravity (item and goal)
        # determine border of item
        # determine pos of border furthest from goal that lies on line
        # calculate unit step towards desired pos (x, y)
        # return: x, y (for where the tcp should be)

        self.update_item_and_goal(item, goal)

        goal_dir = self.get_direction_vector()[0:2] # 2D (x,y)
        item_rot = p.getEulerFromQuaternion(item.ori)[2]

        poke_dir = self.rotate_vector(goal_dir, np.pi)
        item_up_dir = self.rotate_vector(np.asarray([0, item.dim[0]]), item_rot)
        angle_up_poke = self.angle_between_vectors(item_up_dir, poke_dir)

        #print(goal_dir)
        #print(item_up_dir)

        threshold_angle = np.arctan2(item.dim[2], item.dim[0])
        angle_poke = angle_up_poke if angle_up_poke < np.pi else angle_up_poke - np.pi

        # test if dim 0 and 2 are width and height respectively
        length = item.dim[2]
        if 0 <= angle_poke <= threshold_angle:
            angle_poke = angle_poke
            length = item.dim[2]
        elif threshold_angle < angle_poke < np.pi - threshold_angle:
            angle_poke = np.pi / 2 - angle_poke
            length = item.dim[0]
        elif np.pi - threshold_angle <= angle_poke <= np.pi:
            angle_poke = np.pi - angle_poke
            length = item.dim[2]

        dist_center_to_border = length / np.cos(angle_poke)
        #print("dist", dist_center_to_border)

        # Normalize and multiply with calculated distance
        poke_point_vector = (poke_dir / np.linalg.norm(poke_dir)) * dist_center_to_border

        #print("item pos", item.pos)
        p.addUserDebugLine(item.pos, item.pos + [*item_up_dir, 0], [1,0,0], 1, 1)
        p.addUserDebugLine(item.pos, item.pos + [*goal_dir, 0], [1, 1, 1], 1, 1)
        p.addUserDebugLine(item.pos, item.pos + [*poke_point_vector, 0], [0, 0, 0], 1, 1)

        return item.pos[0:2] + poke_point_vector


        #item_right_dir = self.rotate_vector(np.asarray([item.dim[2], 0]), item_rot)
        #item_down_dir = self.rotate_vector(np.asarray([0, -item.dim[0]]), item_rot)
        #item_left_dir = self.rotate_vector(np.asarray([-item.dim[2], 0]), item_rot)
        #angle_right_goal = self.angle_between_vectors(item_right_dir, goal_dir)
        #angle_down_goal = self.angle_between_vectors(item_down_dir, goal_dir)
        #angle_left_goal = self.angle_between_vectors(item_left_dir, goal_dir)
        #print(item_right_dir)
        #print(item_down_dir)
        #print(item_left_dir)
        #print("angle_up_goal", angle_up_goal)
        #print("angle_right_goal", angle_right_goal)
        #print("angle_down_goal", angle_down_goal)
        #print("angle_left_goal", angle_left_goal)
        #p.addUserDebugLine(item.pos, item.pos + [*item_right_dir, 0], [0, 1, 0], 1, 1)
        #p.addUserDebugLine(item.pos, item.pos + [*item_down_dir, 0], [0, 0, 1], 1, 1)
        #p.addUserDebugLine(item.pos, item.pos + [*item_left_dir, 0], [1, 1, 0], 1, 1)