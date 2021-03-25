import pybullet as p
from simulation import *
from expert import *


def main():
    n_steps = 10000

    sim = Simulation()
    expert = Expert()
    dataset = Dataset(n_steps)
    sim.update_state()
    state = sim.get_state()
    #sim.step_to(*(state.item.pos-[0.2, 0]))
    sim.step_to(-0.3, -0.15)
    for i in range(100):
        p.stepSimulation()
        time.sleep(0.02)

    for i in range(n_steps):
        sim.update_state()
        state = sim.get_state()

        poke = expert.calculate_poke2(state.item, state.goal)
        sim.step_to(*poke)
        #  dataset.add(state.image, poke, i)

        p.stepSimulation()
        time.sleep(sim.time_step)

    sim.terminate()
    dataset.save_pokes()


if __name__ == '__main__':
    main()
