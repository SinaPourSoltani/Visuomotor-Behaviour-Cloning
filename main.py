import pybullet as p
from simulation import *
from expert import *


def main():
    sim = Simulation()
    expert = Expert()

    for i in range(10000):
        sim.update_state()
        state = sim.get_state()

        #p.setJointMotorControl2(bodyUniqueId=sim.robotId, jointIndex=1, controlMode=p.POSITION_CONTROL,
        #                        targetPosition=2*math.pi, force=400)

        poke = expert.calculate_poke2(state.item, state.goal)
        print("poke", poke)

        p.stepSimulation()
        time.sleep(sim.time_step / 10)

    sim.terminate()


if __name__ == '__main__':
    main()
