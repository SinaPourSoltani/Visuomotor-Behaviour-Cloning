from utilities import Item
import numpy as np
import pybullet as p


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)


class Expert:
    def __init__(self):
        self.tcp_pose = None
        self.item = None
        self.goal = None
        self.repositioning_tool = False
        self.tcp_poke_angle_threshold = np.pi / 6
        self.tcp_poke_dist_threshold = 0.01
        self.step_size = 0.1       # length of step in meters
        self.approach_dist = 0.03   # distance
        self.work_plane = 0.775
        self.safe_plane = 0.9

    def update_state(self, tcp_pose, item: Item, goal: Item):
        self.tcp_pose = tcp_pose
        self.item = item
        self.goal = goal

    def get_direction_vector(self, from_pos, to_pos):
        dir_vec = to_pos - from_pos
        unit_dir = dir_vec / np.linalg.norm(dir_vec)
        return unit_dir

    def rotate_vector(self, v, a):
        return np.asarray([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])

    def angle_between_vectors(self, v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # def between_points(point1, point2, point_to_check, max_deviation)
    #     # calculate 
    #     dist_to_line = 

    #     if dist_to_line > max_deviation: 
    #         return False
            
    #     # calculate distance to the two points along the direction of the line using trigonometri for a 
    #     dist_point1 = np.norm(point1 - point_to_check
    #     dist_point2 = 

    #     return True 
    

    def calculate_reposition(self, poke_point, poke_dir):
        self.repositioning_tool = True
        poke_point_approach = poke_point + poke_dir * self.approach_dist # poke_point points in the opposite direction of the poke (towards goal).
        move = None

        print("dist2poke", dist(self.tcp_pose[0][0:2], poke_point))
        print("dist2appr", dist(self.tcp_pose[0][0:2], poke_point_approach))

        if dist(self.tcp_pose[0][0:2], poke_point) > self.tcp_poke_dist_threshold and self.tcp_pose[0][2] < self.safe_plane:
            print("ascend")
            move =  np.asarray([0, 0, 1]) * self.step_size
        elif dist(self.tcp_pose[0][0:2], poke_point_approach) < self.tcp_poke_dist_threshold and self.tcp_pose[0][2] > self.work_plane:
            print("descend")
            move = np.asarray([0, 0, -1]) * self.step_size
        elif self.tcp_pose[0][2] >= self.safe_plane:
            print("move to approach")
            vector_step = self.get_direction_vector(self.tcp_pose[0][0:2], poke_point_approach) * self.step_size
            vector_all_the_way = poke_point_approach - self.tcp_pose[0][0:2]

            if np.linalg.norm(vector_step) < np.linalg.norm(vector_all_the_way): 
                move = vector_step
            else:
                move = vector_all_the_way
            move = np.asarray([*move, 0])
        
        if dist(self.tcp_pose[0][0:2], poke_point_approach) < 0.001 and self.tcp_pose[0][2] <= self.work_plane:
            print("reposition done")
            self.repositioning_tool = False
            
        return move

    def calculate_poke2(self, tcp_pose, item: Item, goal: Item):
        # line between centers of gravity (item and goal)
        # determine border of item
        # determine pos of border furthest from goal that lies on line
        # calculate unit step towards desired pos (x, y)
        # return: x, y (relative step for tcp should be)

        self.update_state(tcp_pose, item, goal)

        goal_dir = self.get_direction_vector(item.pos, goal.pos)[0:2] # 2D (x, y)
        poke_dir = self.rotate_vector(goal_dir, np.pi)
        tool_dir = self.get_direction_vector(item.pos, tcp_pose[0])[0:2] # 2D (x, y)

        item_rot = p.getEulerFromQuaternion(item.ori)[2]

        item_up_dir = self.rotate_vector(np.asarray([0, item.dim[0]]), item_rot)
        angle_up_poke = self.angle_between_vectors(item_up_dir, poke_dir)

        box_threshold_angle = np.arctan2(item.dim[2], item.dim[0])
        angle_poke = angle_up_poke if angle_up_poke < np.pi else angle_up_poke - np.pi

        # test if dim 0 and 2 are width and height respectively
        length = item.dim[2]
        if 0 <= angle_poke <= box_threshold_angle:
            angle_poke = angle_poke
            length = item.dim[2]
        elif box_threshold_angle < angle_poke < np.pi - box_threshold_angle:
            angle_poke = np.pi / 2 - angle_poke
            length = item.dim[0]
        elif np.pi - box_threshold_angle <= angle_poke <= np.pi:
            angle_poke = np.pi - angle_poke
            length = item.dim[2]

        dist_center_to_border = (length / 5) / np.cos(angle_poke)
        #print("dist", dist_center_to_border)

        # Normalize and multiply with calculated distance
        poke_point_vector = (poke_dir / np.linalg.norm(poke_dir)) * dist_center_to_border

        #print("item pos", item.pos)
        p.addUserDebugLine(item.pos, item.pos + [*item_up_dir, 0], [1,0,0], 1, 2)
        p.addUserDebugLine(item.pos, item.pos + [*goal_dir, 0], [1, 1, 1], 1, 2)
        p.addUserDebugLine(item.pos, item.pos + [*poke_point_vector, 0], [0, 0, 0], 1, 2)

        poke_point = item.pos[0:2] + poke_point_vector

        if self.repositioning_tool:
            return self.calculate_reposition(poke_point, poke_dir) if not None else np.asarray([*goal_dir, 0]) * self.step_size

        # Only for test = #if dist(tcp_pose[0][0:2], poke_point) < self.tcp_poke_dist_threshold:
        angle_tcp_poke = self.angle_between_vectors(tool_dir, poke_point)
        angle_tcp_poke =  angle_tcp_poke if angle_tcp_poke < np.pi / 2 else np.pi - angle_tcp_poke
        print("angle_tcp_poke", (angle_tcp_poke * 180)/np.pi, "repos", self.repositioning_tool)
        if angle_tcp_poke < self.tcp_poke_angle_threshold: # and self.repositioning_tool == False
            return np.asarray([*goal_dir, 0]) * self.step_size
        else:
            return self.calculate_reposition(poke_point, poke_dir)