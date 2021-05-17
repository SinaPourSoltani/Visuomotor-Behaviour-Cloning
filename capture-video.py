import torchvision.transforms

from simulation import Simulation
from expert import *
from utilities import Dataset, Geometry, get_concat_h_blank, get_concat_v_blank
from BehaviourCloningCNN import get_model
from tqdm import tqdm
import torch
import numpy as np
import os
import sys
import argparse
import cv2

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

    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    img_list = []

    bird_view_matrix = p.computeViewMatrix([1.5, -1.5, 2], [0, 0, 0.8], [0, 0, 1])
    bird_proj_matrix = p.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=5)
    bird_px_width = 448
    bird_px_height = 448


    args = parse_args(args)
    if args.start_idx != 0:
        args.file_mode = 'a'

    sim = Simulation(args.verbose, args.stereo_images)
    expert = Expert(args.verbose)

    if args.test:
        model = get_model(is_stereo=args.stereo_images)
        model.load_state_dict(torch.load("ResNet18_epoch10_baseline_2_0_unfrozen_from_5.pth", map_location=torch.device('cpu')))
        model.eval()
        device = next(model.parameters()).device

    for k in tqdm(range(args.episodes)):
        for i in range(args.MaxSteps):
            state = sim.get_state()

            top_part_of_image = None

            if not args.test:
                tcp_pose = sim.robotArm.get_tcp_pose()
                top_part_of_image = state.image[0].convert('RGB')
                poke = expert.calculate_move(tcp_pose, state.item, state.goal)
            else:
                tf = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                if args.stereo_images:
                    img1 = state.image[0].convert("RGB")
                    img2 = state.image[1].convert("RGB")
                    x1 = tf(img1).unsqueeze_(0).to(device)
                    x2 = tf(img2).unsqueeze_(0).to(device)
                    y = model(x1, x2)
                    top_part_of_image = get_concat_h_blank(img1, img2)
                else:
                    img = state.image.convert("RGB")
                    x = tf(img).unsqueeze_(0).to(device)
                    top_part_of_image = img
                    y = model(x)
                poke = y.cpu().detach().numpy().flatten()
                poke = Geometry.unit_vector(poke) * expert.step_size
                tcp_pose = sim.robotArm.get_tcp_pose()
                poke_for_ori = expert.calculate_move(tcp_pose, state.item, state.goal)
                joined = np.concatenate([poke, poke_for_ori[3:]])

            bird_eye_view = sim.grab_image(bird_view_matrix, bird_proj_matrix, False, bird_px_width, bird_px_height)
            image_collage = get_concat_v_blank(top_part_of_image, bird_eye_view)
            img_list.append(image_collage)

            #sim.set_robot_pose(*joined, mode="rel", useLimits=True)
            sim.set_robot_pose(*poke, mode="rel", useLimits=True)
            sim.step(False)

            if expert.STATE == ON_GOAL:
                break

        sim.reset_environment()
    
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (448, 672))

    for i in range(len(img_list)):
        out.write(np.array(img_list[i])[:,:,::-1])
    out.release()

    print("Done!\nTerminating...")
    sim.terminate()

if __name__ == '__main__':
    main()
