from simulation import Simulation
from expert import Expert
from utilities import Dataset
import os 
import sys
import argparse

def parse_args(args): 
    parser = argparse.ArgumentParser(description="Simple Script for generating training examples for a visuomotor task on a 2D-plane")
    
    parser.add_argument('--image_path', help="Path to where images should be saved", default="data/images/", type=str)
    parser.add_argument('--file_mode', help="Mode of the data file w: create new file, potentially overwrite, a: append to file existing, x: only create file if it it does not exists",default="w" , type=str)
    parser.add_argument('--data_file_name', help="Name of the file where tabular data is stored", default="test.npy", type=str)
    return parser.parse_args(args)

def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    print("args", args)

    n_steps = 10000

    sim = Simulation()
    expert = Expert()
    dataset = Dataset(args.data_file_name, image_path=args.image_path,  filemode=args.file_mode)
    sim.update_state()
    state = sim.get_state()
    #sim.step_to(*(state.item.pos-[0.2, 0])

    sim.step_robot_to(-0.3, -0.15)
    for _ in range(100):
        sim.step_simulation()

    for i in range(n_steps):
        state = sim.get_state()
        poke = expert.calculate_poke2(state.item, state.goal)
        dataset.add(state.image, poke, i)
        sim.step_robot_to(*poke)
        

        sim.step_simulation()

    sim.terminate()



if __name__ == '__main__':
    main()
