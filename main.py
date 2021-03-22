import pybullet as p
from simulation import *
from expert import *

def main():

    sim = Simulation()

    for i in range(10000):
        sim.update_state()
        state = sim.get_state()

        sim.step_to(0.0, -0.1)

        # expert.update_item_and_goal(state.item, state.goal)
        # poke = expert.calculate_poke()

        p.stepSimulation()
        time.sleep(sim.time_step)

    sim.terminate()


if __name__ == '__main__':
    main()

















