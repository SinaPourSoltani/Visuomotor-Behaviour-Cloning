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
    parser.add_argument('--start_idx', help="Start index of episodes", default=0, type=int)
    parser.add_argument('--verbose', help="Print out stuff while executing", action='store_true')
    parser.add_argument('--episodes', help="Number of episodes to be contained in the dataset",default=1000, type=int )
    parser.add_argument('--MaxSteps', help="Maximum number of step in one episode", default=10000, type=int)
    return parser.parse_args(args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)
    if args.start_idx != 0:
        args.file_mode = 'a'
    sim = Simulation(args.verbose)

    expert = Expert(args.verbose)
    dataset = Dataset(args.verbose, args.data_file_name, image_path=args.image_path,  filemode=args.file_mode, start_idx=args.start_idx)

    sim.set_robot_pose(-0.25, -0.15, 0.775)
    for _ in range(args.episodes):
        for _ in range(args.MaxSteps):
            state = sim.get_state()

            tcp_pose = sim.robotArm.get_tcp_pose()
            sim.draw_coordinate_frame(*tcp_pose)
            poke = expert.calculate_move(tcp_pose, state.item, state.goal)
            dataset.add(state.image, poke)

            sim.set_robot_pose(*poke, mode="rel", useLimits=True)
            sim.step(False)


            if expert.STATE == 105: 
                break

        dataset.next_episode()
        sim.reset_environment()



    #dataset.next_episode()
    print("Done!\nTerminating...")

    sim.terminate()



if __name__ == '__main__':
    main()
