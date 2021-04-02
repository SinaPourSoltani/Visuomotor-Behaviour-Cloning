from simulation import Simulation
from expert import Expert
from utilities import Dataset

def main():

    n_steps = 10000

    sim = Simulation()
    expert = Expert()
    dataset = Dataset("data/test.npy", image_path="data/images/", filemode="w")
    sim.update_state()
    state = sim.get_state()
    #sim.step_to(*(state.item.pos-[0.2, 0])

    sim.set_robot_pose(-0.3, -0.15)
    for _ in range(100):
        sim.step()
    

    for i in range(n_steps):
        state = sim.get_state()

        tcp_pose = sim.robotArm.get_tcp_pose()
        sim.draw_coordinate_frame(*tcp_pose)
        poke = expert.calculate_poke2(tcp_pose, state.item, state.goal)
        print(tcp_pose[0], ' + ', poke)
        
        #dataset.add(state.image, poke, i)
        sim.set_robot_pose(*poke, mode='rel')

        sim.step(False)

    sim.terminate()
   # dataset.save_pokes()


if __name__ == '__main__':
    main()
