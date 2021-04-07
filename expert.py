from utilities import Item, Geometry as geo
import numpy as np
import pybullet as p


CALC_POKE = 101
ASCEND = 102
MOVE_APPROACH = 103
DESCEND = 104
ON_GOAL = 105

state2str = {
    CALC_POKE: "CALC_POKE",
    ASCEND: "ASCEND",
    MOVE_APPROACH: "MOVE_APPROACH",
    DESCEND: "DESCEND",
    ON_GOAL: "ON_GOAL",
}


class Expert:
    def __init__(self):
        # State machine
        self.STATE = CALC_POKE

        # Geometry
        self.tcp_pose = None    # TCP Pose [pos, ori]
        self.item = None        # Of type Item from utilities
        self.goal = None        # of type Item from utilities

        self.goal_dir = None    # Direction of goal position wrt. item pos
        self.poke_dir = None    # Direction of poke point wrt. item pos
        self.tool_dir = None    # Direction of TCP wrt. item pos
        self.poke_point = None  # Position of poke point wrt. world

        # Thresholds
        # TODO: Adjust/tune thresholds
        self.safe_plane = 0.9
        self.work_plane = 0.775

        self.tcp_poke_angle_threshold = np.pi / 6
        self.tcp_poke_dist_threshold = 0.01
        self.tcp_approach_dist_threshold = 0.03
        self.tcp_goal_line_dist_threshold = 0.10
        self.item_goal_dist_threshold = 0.05

        self.step_size = 0.1

    def calculate_poke_point(self):
        item_rot = p.getEulerFromQuaternion(self.item.ori)[2]

        item_up_dir = geo.rotate_vector(np.asarray([0, self.item.dim[0]]), item_rot)
        angle_up_poke = geo.angle_between_vectors(item_up_dir, self.poke_dir)

        box_threshold_angle = np.arctan2(self.item.dim[2], self.item.dim[0])
        angle_poke = angle_up_poke if angle_up_poke < np.pi else angle_up_poke - np.pi

        # TODO: test if dim 0 and 2 are width and height respectively
        length = self.item.dim[2]
        if 0 <= angle_poke <= box_threshold_angle:
            angle_poke = angle_poke
            length = self.item.dim[2]
        elif box_threshold_angle < angle_poke < np.pi - box_threshold_angle:
            angle_poke = np.pi / 2 - angle_poke
            length = self.item.dim[0]
        elif np.pi - box_threshold_angle <= angle_poke <= np.pi:
            angle_poke = np.pi - angle_poke
            length = self.item.dim[2]

        dist_center_to_border = (length / 5) / np.cos(angle_poke)
        poke_point_vector = (self.poke_dir / np.linalg.norm(self.poke_dir)) * dist_center_to_border
        return self.item.pos[0:2] + poke_point_vector

    def update_geometry(self, tcp_pose, item: Item, goal: Item):
        self.tcp_pose = tcp_pose
        self.item = item
        self.goal = goal

        self.goal_dir = geo.get_direction_vector(self.item.pos, self.goal.pos)[0:2] # 2D (x, y)
        self.poke_dir = geo.rotate_vector(self.goal_dir, np.pi)
        self.tool_dir = geo.get_direction_vector(self.item.pos, self.tcp_pose[0])[0:2] # 2D (x, y)
        self.poke_point = self.calculate_poke_point()

    def calculate_move(self, tcp_pose, item: Item, goal: Item):
        self.update_geometry(tcp_pose, item, goal)

        print(state2str.get(self.STATE, None))

        if self.STATE == CALC_POKE:
            move = np.asarray([*self.goal_dir, 0])

            if geo.distance_to_line(self.tcp_pose[0][0:2], self.item.pos[0:2], self.goal_dir) > self.tcp_goal_line_dist_threshold or geo.dist(self.tcp_pose[0][0:2], self.goal.pos[0:2]) < geo.dist(self.item.pos[0:2], self.goal.pos[0:2]):
                self.STATE = ASCEND

            if geo.dist(self.item.pos, self.goal.pos) <= self.item_goal_dist_threshold:
                self.STATE = ON_GOAL
                move = np.asarray([0, 0, 0])


        elif self.STATE == ASCEND:
            move = np.asarray([0, 0, 1])

            print("tcp_z", self.tcp_pose[0][2], self.tcp_pose[0][2] >= self.safe_plane)
            if self.tcp_pose[0][2] >= self.safe_plane:
                self.STATE = MOVE_APPROACH


        elif self.STATE == MOVE_APPROACH:
            poke_point_approach = self.poke_point + self.poke_dir * self.tcp_approach_dist_threshold

            vector_step = geo.get_direction_vector(self.tcp_pose[0][0:2], poke_point_approach)
            vector_all_the_way = poke_point_approach - self.tcp_pose[0][0:2]

            # Determine if approach is closer than a step size
            move = vector_step if np.linalg.norm(vector_step) < np.linalg.norm(vector_all_the_way) else vector_all_the_way / self.step_size
            move = np.asarray([*move, 0])

            print("dist_tcp_approach_point", geo.dist(self.tcp_pose[0][0:2], poke_point_approach), geo.dist(self.tcp_pose[0][0:2], poke_point_approach) < self.tcp_approach_dist_threshold)
            if geo.dist(self.tcp_pose[0][0:2], poke_point_approach) < self.tcp_approach_dist_threshold:
                self.STATE = DESCEND


        elif self.STATE == DESCEND:
            move = np.asarray([0, 0, -1])

            print("tcp_z", self.tcp_pose[0][2], self.tcp_pose[0][2] <= self.work_plane)
            if self.tcp_pose[0][2] <= self.work_plane:
                self.STATE = CALC_POKE


        elif self.STATE == ON_GOAL:
            move = np.asarray([0, 0, 0])

            print("dist_item_goal", geo.dist(self.item.pos, self.goal.pos), geo.dist(self.item.pos, self.goal.pos) > self.item_goal_dist_threshold)
            if geo.dist(self.item.pos, self.goal.pos) > self.item_goal_dist_threshold:
                self.STATE = CALC_POKE

        else:
            print("Something went wrong. STATE unknown.")

        return move * self.step_size


