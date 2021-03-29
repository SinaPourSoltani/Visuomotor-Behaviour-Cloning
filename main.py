from simulation import Simulation
from expert import Expert
from utilities import Dataset

def main():

    n_steps = 10000

    sim = Simulation()
    expert = Expert()
    dataset = Dataset(n_steps)
    sim.update_state()
    state = sim.get_state()
    #sim.step_to(*(state.item.pos-[0.2, 0])

    sim.step_robot_to(-0.3, -0.15)
    for _ in range(100):
        sim.step_simulation()

    for _ in range(n_steps):
        state = sim.get_state()
        poke = expert.calculate_poke2(state.item, state.goal)
        sim.step_robot_to(*poke)
        #  dataset.add(state.image, poke, i)

        sim.step_simulation()

    sim.terminate()
    dataset.save_pokes()


if __name__ == '__main__':
    main()
