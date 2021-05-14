import pybullet as p
import numpy as np
import time
import math
import pybullet_data
from utilities import *
from ur5 import load_arm_dim_up
#from utilities import State



class Simulation:
    def __init__(self, verbose, stereo_images):
        # Parser arguments
        self.verbose = verbose
        self.stereo_images = stereo_images

        self.physicsClient = p.connect(p.GUI)
        self.state = None


        self.p = p
        self.p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,1)

        # Physics
        self.gravity = [0, 0, -9.82]
        self.move_precision = 0.001
        self.time_step = 1./240.

        # Objects
        self.tableId = None
        self.robotArm = None
        self.itemId = None
        self.goalId = None
        self.minGoalObj_dist = 0.1

        # Camera
        self.px_width = 224
        self.px_height = 224
            # Mono
        self.view_matrix = None
        self.proj_matrix = None
            # Stereo
        self.view_matrix_left = None
        self.proj_matrix_left = None
        self.view_matrix_right = None
        self.proj_matrix_right = None



        self.setup_environment()
        self.reset_environment()

    #cam_pos = [0, -1.5, 2], target_pos = [0, 0, 0.8], cam_up_pos = [0, 0, 1]
    #left_cam : cam_pos=[0.9, -0.7, 1.6], target_pos=[-0.2, 0.2, 0.5], cam_up_pos=[0, 0, 1]
    #right_cam : cam_pos=[0.9, 0.7, 1.6], target_pos=[-0.2, -0.2, 0.5], cam_up_pos=[0, 0, 1]
    def setup_camera(self):
        cam_up_pos=[0, 0, 1]
        if not self.stereo_images:
            self.view_matrix = p.computeViewMatrix([0, -1.5, 2], [0, 0, 0.8], cam_up_pos)
            self.proj_matrix = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=5)
        else:
            self.view_matrix_left = p.computeViewMatrix([0, -1.5, 2], [0, 0, 0.8], cam_up_pos)
            self.proj_matrix_left = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=5)
            self.view_matrix_right = p.computeViewMatrix([1.3, 0, 2], [0, 0, 0.8], cam_up_pos)
            self.proj_matrix_right = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=5)

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


        object_constraints = [(-0.24, 0.24), (-0.35, 0.35), 0.6567, 0, 0, (-math.pi, math.pi)]
        goal_constraints = [(-0.20, 0.20), (-0.25, 0.25), 0.633, 0, 0, (-math.pi, math.pi)]


        object_pose = self.random_pose(constraints=object_constraints)
        #object_pose = [[-0.25, 0.35, 0.6567], [0, 0, math.pi/4]]
        while True:     
            goal_pose = self.random_pose(constraints=goal_constraints)
            dist_poses =  Geometry.dist(goal_pose[0][:2], object_pose[0][:2])
            if dist_poses > self.minGoalObj_dist:
                break
   
        if self.verbose:
            print(object_pose)
            print(goal_pose)

        self.goalId = p.loadURDF('objects/goal/lego.urdf', goal_pose[0], p.getQuaternionFromEuler(goal_pose[1]), globalScaling=3, useFixedBase=True)

        self.itemId = p.loadURDF('objects/lego/lego.urdf', object_pose[0], p.getQuaternionFromEuler(object_pose[1]), globalScaling=3)
        #self.itemId = p.loadURDF('objects/cylinder/cylinder.urdf', object_pose[0], p.getQuaternionFromEuler(object_pose[1]), globalScaling=3)

    def setup_environment(self):
        p.setGravity(*self.gravity)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)

        self.tableId = p.loadURDF("objects/table/table.urdf", [0, 0, 0])
        self.robotArm = load_arm_dim_up('ur5', self.verbose, dim='Z')

        self.setup_camera()

    def reset_environment(self):
        # reset the position of the robot
        self.robotArm.resetJointPoses()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.time_step)
        self.set_random_object_and_goal()

        self.update_state()

    def grab_image(self, view_matrix, proj_matrix, show=False):
        (_, _, px, _, _) = p.getCameraImage(self.px_width,self.px_height, view_matrix, proj_matrix, shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img_data = np.array(px)
        img = Image.fromarray(img_data, 'RGBA').convert('RGB')
        if show:
            img.show()
            input("Press enter to continue")
        return img

    def update_state(self):
        image = None
        if not self.stereo_images:
            image = self.grab_image(self.view_matrix, self.proj_matrix)#[self.grab_image(self.view_matrix, self.proj_matrix)]
        else:
            image_l = self.grab_image(self.view_matrix_left, self.proj_matrix_left)
            image_r = self.grab_image(self.view_matrix_right, self.proj_matrix_right)
            image = [image_l, image_r]

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

    def set_robot_pose(self, x, y, z, ori_x=0, ori_y=1/2*math.pi, ori_z=0, finger_angle=1.3, mode='abs', precision=0.001, useLimits=False):
        i = 0
        max_sim_steps = 500
        ori = p.getQuaternionFromEuler([ori_x, ori_y, ori_z])

        if mode == 'rel':
            pose = self.robotArm.get_tcp_pose()
            tcp_x, tcp_y, tcp_z = pose[0]
            x += tcp_x
            y += tcp_y
            z += tcp_z


        # define our limits.
        if useLimits:
            z = max(0.775, z)
            #x = max(-0.25, min(0.3, x))
            #y = max(-0.4, min(0.4, y))

        if self.verbose: print("pos z: ", z)
        for i in range (max_sim_steps):
            #print("Current step: ", i + 1)
            pose = self.robotArm.get_tcp_pose()
            tcp_x, tcp_y, tcp_z = pose[0]
            dist = math.sqrt((x-tcp_x)**2 + (y-tcp_y)**2 + (z-tcp_z)**2)
            if dist < precision:
                if self.verbose: print("Pos reached! step: ", i)
                break
            self.robotArm.move_to(x, y, z, ori, finger_angle)
            self.step(False)

        #if i == max_sim_steps - 1:
            #print("Max simulation steps reached! You fucked up!")
        self.update_state()






