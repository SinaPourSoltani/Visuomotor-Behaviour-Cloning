import torchvision.transforms

from simulation import Simulation
from expert import *
from utilities import Dataset, Geometry
from BehaviourCloningCNN import get_model
from tqdm import tqdm
import torch
import numpy as np
import os
import sys
import argparse
from PIL import Image

def parse_args(args):
    parser = argparse.ArgumentParser(description="Simple Script for generating training examples for a visuomotor task on a 2D-plane")

    parser.add_argument('--image_path', help="Path to where images should be saved", default="data/images/", type=str)
    parser.add_argument('--data_file_path', help="", default="data/", type=str)
    parser.add_argument('--file_mode', help="Mode of the data file w: create new file, potentially overwrite, a: append to file existing, x: only create file if it it does not exists",default="w" , type=str)
    parser.add_argument('--data_file_name', help="Name of the file where tabular data is stored", default="test.csv", type=str)
    parser.add_argument('--start_idx', help="Start index of episodes", default=0, type=int)
    parser.add_argument('--verbose', help="Print out stuff while executing", action='store_true')
    parser.add_argument('--episodes', help="Number of episodes to be contained in the dataset",default=500, type=int )
    parser.add_argument('--MaxSteps', help="Maximum number of step in one episode", default=500, type=int)
    parser.add_argument('--test', help="Set whether to gather data with the expert or test with the model",default=False, type=bool)
    parser.add_argument('--stereo_images', help="Set whether to use a stereo camera setup or a mono setup", default=False, type=bool)
    parser.add_argument('--model_name', help="Name of the model to be used", type=str)
    parser.add_argument('--complex_mlp', help="Whether to use a complex model", default=False, type=bool)

    return parser.parse_args(args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)
    if args.start_idx != 0:
        args.file_mode = 'a'
    sim = Simulation(args.verbose, args.stereo_images)
    succes = 0
  
    print("test:", args.test)
    expert = Expert(args.verbose)
    dataset = Dataset(args.verbose, args.stereo_images, args.data_file_name, image_path=args.image_path, data_file_path=args.data_file_path, filemode=args.file_mode, start_idx=args.start_idx)

    if args.test:
        #def get_model(normalise_poke_vec=False, complex_mlp=False, is_stereo=False, p_dropout=0):
        model = get_model(complex_mlp=args.complex_mlp, is_stereo=args.stereo_images)
        model.load_state_dict(torch.load(args.model_name))#, map_location=torch.device('cpu')))
        model.eval()
        device = next(model.parameters()).device
        #print(device)

    t = tqdm(range(args.episodes))
    for episode in t:
        for _ in range(args.MaxSteps):
            state = sim.get_state()

            if not args.test:
                tcp_pose = sim.robotArm.get_tcp_pose()
                #sim.draw_coordinate_frame(*tcp_pose)
                poke = expert.calculate_move(tcp_pose, state.item, state.goal)
                #print("exportPoke")
                #dataset.add(state.image, poke)
            else:

                tf = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

                if args.stereo_images:
                    img1 = state.image[0].convert("RGB")
                    img2 = state.image[1].convert("RGB")
                    x1 = tf(img1).unsqueeze_(0).to(device)
                    x2 = tf(img2).unsqueeze_(0).to(device)
                    y = model(x1, x2)

                else:
                    img = state.image.convert("RGB")
                    x = tf(img).unsqueeze_(0).to(device)
                    y = model(x)

                poke = y.cpu().detach().numpy().flatten()
                poke = Geometry.unit_vector(poke) * expert.step_size
                #print(poke)
                tcp_pose = sim.robotArm.get_tcp_pose()
                poke_for_ori = expert.calculate_move(tcp_pose, state.item, state.goal)

                joined = np.concatenate([poke, poke_for_ori[3:]]) # why??


            sim.set_robot_pose(*joined, mode="rel", useLimits=True)
            #sim.set_robot_pose(*poke, mode="rel", useLimits=True)
            sim.step(False)

            if expert.STATE == ON_GOAL:
                succes += 1
                break
        t.set_description(f'Succes: {succes} | Succes Rate: {succes / (episode + 1):.4f}')

        dataset.next_episode()
        sim.reset_environment()



    #dataset.next_episode()
    print("Succes: {} | Succes rate: {:.4f}%".format(succes, 100*succes/args.episodes))
    print("Done!\nTerminating...")

    sim.terminate()



if __name__ == '__main__':
    main()
