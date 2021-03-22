import pybullet as p
from simulation import *
from expert import *

def main():

    n_steps = 10000

    sim = Simulation()
    expert = Expert()
    dataset = Dataset(n_steps)


    for i in range(n_steps):
        sim.update_state()
        state = sim.get_state()


        poke = expert.calculate_poke2(state.item, state.goal)
        dataset.add(state.image, poke, i)

        p.stepSimulation()
        time.sleep(sim.time_step)

    sim.terminate()
    dataset.save_pokes()


if __name__ == '__main__':
    main()
