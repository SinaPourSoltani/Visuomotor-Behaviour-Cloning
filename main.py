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
    parser.add_argument('--data_file_name', help="Name of the file where tabular data is stored", default="test.csv", type=str)
    parser.add_argument('--verbose', help="Print out stuff while executing", action='store_true')
    return parser.parse_args(args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)

    n_steps = 10000

    sim = Simulation(args.verbose)
    expert = Expert(args.verbose)

    dataset = Dataset(args.verbose, args.data_file_name, image_path=args.image_path,  filemode=args.file_mode)
    sim.update_state()
    state = sim.get_state()


    sim.set_robot_pose(-0.25, -0.15, 0.775)

    for i in range(n_steps):
        state = sim.get_state()

        tcp_pose = sim.robotArm.get_tcp_pose()
        sim.draw_coordinate_frame(*tcp_pose)
        poke = expert.calculate_move(tcp_pose, state.item, state.goal)
        #print(tcp_pose[0], ' + ', poke)

        #dataset.add(state.image, poke, i)
        sim.set_robot_pose(*poke, mode="rel", useLimits=True)

        sim.step(False)
        if expert.STATE == 105:
            break

    print("Done!\nTerminating...")
    while True:
        pass


    sim.terminate()



if __name__ == '__main__':
    main()
