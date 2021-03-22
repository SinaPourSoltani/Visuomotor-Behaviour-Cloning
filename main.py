import pybullet as p
from simulation import *
from expert import *

def main():

    sim = Simulation()
    # reset the position of the robot
    sim.robotId.resetJointPoses()
    for i in range(100):
        p.stepSimulation()
        time.sleep(sim.time_step * 10)

    # z 0.775 puts the tool relative close to the surface, when tool is turned 90 degrees around y-axis
    targetPos = [-0.0, -0.1, 0.775]
    targetOri = [ 0, 1/2*math.pi, 0] # point the end effector downward

    action = targetPos + list(p.getQuaternionFromEuler(targetOri)) + [1.2]

    for i in range(10000):
        sim.update_state()
        state = sim.get_state()

        sim.step_to(action)

        # expert.update_item_and_goal(state.item, state.goal)
        # poke = expert.calculate_poke()

        p.stepSimulation()
        time.sleep(sim.time_step)

    sim.terminate()


if __name__ == '__main__':
    main()

















