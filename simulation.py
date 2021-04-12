import pybullet as p
import numpy as np
import time
import math
import pybullet_data
from utilities import *
from ur5 import load_arm_dim_up
#from utilities import State



class Simulation:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        self.state = None

        self.p = p

        # Physics
        self.gravity = [0, 0, -9.82]
        self.move_precision = 0.001
        self.time_step = 1./240.

        # Objects
        self.tableId = None
        self.robotArm = None
        self.itemId = None
        self.goalId = None

        # Camera
        self.px_width = 600
        self.px_height = 600
        self.view_matrix = None
        self.proj_matrix = None

        self.setup_environment()
        self.reset_environment()

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

        constraints = [(-0.3, 0.3), (-0.3, 0.3), 0.6367, 0, 0, (-math.pi, math.pi)]
        object_constraints = [-0.1, -0.1, 0.6367, 0, 0, math.pi/4]
        goal_constraints = [0.2, -0.1, 0.6251, 0, 0, 0]

        object_pose = self.random_pose(constraints=constraints)
        goal_pose = self.random_pose(constraints=constraints)
        print(object_pose)
        print(goal_pose)

        self.goalId = p.loadURDF('objects/goal/lego.urdf', goal_pose[0], p.getQuaternionFromEuler(goal_pose[1]), globalScaling=3, useFixedBase=True)
        self.itemId = p.loadURDF('objects/lego/lego.urdf', object_pose[0], p.getQuaternionFromEuler(object_pose[1]), globalScaling=3)

    def setup_environment(self):
        p.setGravity(*self.gravity)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)

        self.tableId = p.loadURDF("objects/table/table.urdf", [0, 0, 0])
        self.robotArm = load_arm_dim_up('ur5', dim='Z')


        self.set_random_object_and_goal()
        self.setup_camera(cam_pos=[0, -1.5, 2], target_pos=[0, 0, 0.8])

    def reset_environment(self):
        # reset the position of the robot
        self.robotArm.resetJointPoses()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.time_step)

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

    def draw_coordinate_frame(self, pos, ori=[0,0,0,1]):
        rot = p.getMatrixFromQuaternion(ori)
        colors = [(0,0,1), (0,1,0), (1,0,0)]
        for i, el in enumerate(pos):
            basis_vec = rot[i*3 : (i+1)*3]
            coord_line = [a + b for a, b in zip(pos, basis_vec)]
            p.addUserDebugLine(pos, coord_line, colors[i], 1, 2)

    def step(self, sleep=False): 
        p.stepSimulation()
        if sleep: 
            time.sleep(self.time_step)

    def set_robot_pose(self, x, y, z, ori=[ 0, 1/2*math.pi, 0], finger_angle=1.3, mode='abs', precision=0.001, useLimits=False):
        i = 0
        max_sim_steps = 500
        ori = p.getQuaternionFromEuler(ori)

        if mode == 'rel':
            pose = self.robotArm.get_tcp_pose()
            tcp_x, tcp_y, tcp_z = pose[0]
            x += tcp_x
            y += tcp_y
            z += tcp_z
            ori += pose[1]

        # define our limits.
        if useLimits:
            z = max(0.775, z)
            # x = max(-0.25, min(0.3, x))
            # y = max(-0.4, min(0.4, y))

        print("pos z: ", z)
        for i in range (max_sim_steps):
            #print("Current step: ", i + 1)
            pose = self.robotArm.get_tcp_pose()
            tcp_x, tcp_y, tcp_z = pose[0]
            dist = math.sqrt((x-tcp_x)**2 + (y-tcp_y)**2 + (z-tcp_z)**2)
            if dist < precision:
                print("Pos reached! step: ", i)
                break
            self.robotArm.move_to(x, y, z, ori, finger_angle)
            self.step(False)

        #if i == max_sim_steps - 1:
            #print("Max simulation steps reached! You fucked up!")
        self.update_state()






