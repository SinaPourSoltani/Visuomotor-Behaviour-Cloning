import pybullet as p
import numpy as np
import pybullet_data
from expert import *
import os

from collections import namedtuple
from attrdict import AttrDict


def setup_sisbot(p, uid):
    controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint", 'left_gripper_motor', 'right_gripper_motor']

    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(uid)
    jointInfo = namedtuple("jointInfo",
                           ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(uid, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
                         jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
        if info.type=="REVOLUTE": # set revolute joint to static
            p.setJointMotorControl2(uid, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    controlRobotiqC2 = False
    mimicParentName = False
    return joints, controlRobotiqC2, controlJoints, mimicParentName

def load_arm_dim_up(arm, dim = 'Z'):
    arm = ur5()
    if dim == 'Y':
        arm_rot = p.getQuaternionFromEuler([-np.pi / 2, (1 / 2) * np.pi, 0])
        arm.setPosition([0, -0.1, 0.5], [arm_rot[0], arm_rot[1], arm_rot[2], arm_rot[3]])
    else:
        arm_rot = p.getQuaternionFromEuler([0, 0, 0])#-1/2*math.pi])
        arm.setPosition([-0.5, 0.0, 0.525], [arm_rot[0], arm_rot[1], arm_rot[2], arm_rot[3]])
    return arm

class ur5:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath()):

        self.robotUrdfPath = "objects/urdf/real_arm.urdf"

        self.robotStartPos = [0.0,0.0,0.0]
        self.robotStartOrn = p.getQuaternionFromEuler([1.885,1.786,0.132])
        self.tcpDefaultOri = p.getQuaternionFromEuler([ 0, 1/2*np.pi, 0]) # point the end effector downward


        self.xin = self.robotStartPos[0]
        self.yin = self.robotStartPos[1]

        self.zin = self.robotStartPos[2]
        self.active = False
        self.lastJointAngle = None

        self.reset()

    def reset(self):
        print("----------------------------------------")
        print("Loading robot from {}".format(self.robotUrdfPath))
        self.uid = p.loadURDF(os.path.join(os.getcwd(), self.robotUrdfPath), self.robotStartPos, self.robotStartOrn,
                              flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlRobotiqC2, self.controlJoints, self.mimicParentName = setup_sisbot(p, self.uid)
        self.endEffectorIndex = 7  # ee_link
        self.numJoints = p.getNumJoints(self.uid)
        self.active_joint_ids = []
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            self.active_joint_ids.append(joint.id)

    def setPosition(self, pos, quat):
        p.resetBasePositionAndOrientation(self.uid, pos,
                                          quat)

    def resetJointPoses(self):

        # move to this ideal init point
        self.active = False

        for i in range(0, 50000):
            self.action([0.15328961509984124, -1.8, -1.5820032364177563, -1.2879050862601897, 1.5824233979484994,
                         0.19581299859677043, 0.012000000476837159, -0.012000000476837159])
        self.active = True
        self.lastJointAngle = [0.15328961509984124, -1.8, -1.5820032364177563, -1.2879050862601897, 1.5824233979484994,
                               0.19581299859677043]

    def action(self, motorCommands):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]

            poses.append(motorCommands[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)
        l = len(poses)

        p.setJointMotorControlArray(self.uid, indexes, p.POSITION_CONTROL, targetPositions=poses)#,
        #                             #positionGains=[0.06] * l, forces=forces)
        #for i in range(6):
        #    p.resetJointState(self.uid, indexes[i], poses[i])
        #targetVelocities=[0] * l,
        #p.setJointMotorControlArray(self.uid indexes, p.POSITION_CONTROL, targetPositions=poses, positionGains=[0.03] * l, forces=forces)
        # holy shit this is so much faster in arrayform!

    def get_tcp_pose(self):
        (pos, ori, _, _, _, _) = p.getLinkState(self.uid, self.endEffectorIndex, computeForwardKinematics=1)
        return np.asarray(pos), np.asarray(ori)

    def move_to(self, x, y, z, ori, finger_angle, mode='abs', useLimits = False):
        # z 0.775 puts the tool relative close to the surface, when tool is turned 90 degrees around y-axis
        ori = p.getQuaternionFromEuler(ori)

        if mode == 'rel':
            pose = self.get_tcp_pose()
            tcp_x, tcp_y, tcp_z = pose[0]
            x += tcp_x
            y += tcp_y
            z += tcp_z
            ori += pose[1]
        
        # define our limits.
        if useLimits:
            z = max(0.14, min(0.7, z))
            x = max(-0.25, min(0.3, x))
            y = max(-0.4, min(0.4, y))

        jointPose = list(p.calculateInverseKinematics(self.uid, self.endEffectorIndex, [x, y, z], ori))

        jointPose[7] = -finger_angle / 25
        jointPose[6] = finger_angle / 25

        self.action(jointPose)

            #p.setJointMotorControlArray(self.uid, indexes, p.POSITION_CONTROL, targetPositions=poses)#,
        # print(jointPose)
        return jointPose

