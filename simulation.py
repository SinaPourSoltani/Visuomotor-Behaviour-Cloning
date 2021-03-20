import pybullet as p
import numpy as np
import time
import math
import pybullet_data
from utilities import *


class Simulation:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        self.state = None

        # Physics
        self.gravity = [0, 0, -9.82]
        self.time_step = 1./240.

        # Objects
        self.tableId = None
        self.robotId = None
        self.itemId = None
        self.goalId = None

        # Camera
        self.px_width = 600
        self.px_height = 600
        self.view_matrix = None
        self.proj_matrix = None

        self.setup_environment()

    def setup_camera(self, cam_pos=[0, -1.5, 2], target_pos=[0, 0, 0.8], cam_up_pos=[0, 0, 1]):
        self.view_matrix = p.computeViewMatrix(cam_pos, target_pos, cam_up_pos)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=5)

    def random_pose(self, constraints):
        pose = []
        for el in constraints:
            if type(el) is not tuple:
                pose.append(el)
            else:
                r = np.random.uniform(el[0], el[1])
                pose.append(r)
        return pose[:3], pose[3:]

    def set_random_object_and_goal(self):
        if self.itemId is not None:
            p.removeBody(self.itemId)
        if self.goalId is not None:
            p.removeBody(self.goalId)

        #constraints = [(-0.3, 0.3), (-0.3, 0.3), 0.6367, 0, 0, (-math.pi, math.pi)]
        object_constraints = [0, 0, 0.6367, 0, 0, 0]
        goal_constraints = [0.2, 0, 0.6251, 0, 0, 0]

        object_pose = self.random_pose(constraints=object_constraints)
        goal_pose = self.random_pose(constraints=goal_constraints)
        print(object_pose)
        print(goal_pose)

        self.goalId = p.loadURDF('goal/lego.urdf', goal_pose[0], p.getQuaternionFromEuler(goal_pose[1]), globalScaling=3, useFixedBase=True)
        self.itemId = p.loadURDF('lego/lego.urdf', object_pose[0], p.getQuaternionFromEuler(object_pose[1]), globalScaling=3)

    def setup_environment(self):
        p.setGravity(*self.gravity)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)

        p.setAdditionalSearchPath('./objects/')
        self.tableId = p.loadURDF("table/table.urdf", [0, 0, 0])
        self.robotId = p.loadURDF("ur3_with_gripper/ur3_with_gripper.urdf", [0, 0.4, 0.625], p.getQuaternionFromEuler([0, 0, math.pi]))

        self.set_random_object_and_goal()
        self.setup_camera(cam_pos=[0, -1.5, 2], target_pos=[0, 0, 0.8])

    def grab_image(self, show=False):
        (_, _, px, _, _) = p.getCameraImage(self.px_width,self.px_height, self.view_matrix, self.proj_matrix)
        img_data = np.array(px)
        img = Image.fromarray(img_data, 'RGBA')
        if show:
            img.show()
            input("Press enter to continue")
        return img

    def update_state(self):
        image = self.grab_image()
        item = Item([], [], [])
        goal = Item([], [], [])

        item_pose = p.getBasePositionAndOrientation(self.itemId)
        item_dim = p.getVisualShapeData(self.itemId)[0][3]

        goal_pose = p.getBasePositionAndOrientation(self.goalId)
        goal_dim = p.getVisualShapeData(self.goalId)[0][3]

        item.set_pos_and_ori_from_pose(item_pose)
        item.set_dim(item_dim)
        goal.set_pos_and_ori_from_pose(goal_pose)
        goal.set_dim(goal_dim)

        self.state = State(image, item, goal)

    def get_state(self):
        return self.state

    def terminate(self):
        p.disconnect()
