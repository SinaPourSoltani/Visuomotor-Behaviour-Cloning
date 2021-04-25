import os
from zipfile import ZipFile
import copy
import numpy as np
import pandas as pd
import csv
import random
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, transforms

def load_data(data_link):
    if not os.path.exists('data'):
      !wget data_link
      with ZipFile('images.zip', 'r') as z:
        z.extractall('data/')
      !mv 'data/images/test.csv' 'data/test.csv'

class PokeDataset(Dataset):
  def __init__(self, csv_file_path, image_folder_path, episode_indeces, transforms=None):
    self.image_folder_path = image_folder_path
    self.transforms = transforms
    self.poke_frame = pd.read_csv(csv_file_path)
    self.poke_frame = self.poke_frame.loc[self.poke_frame['episode'].isin(episode_indeces)]

    self.transforms = transforms

  def __len__(self):
    return len(self.poke_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    image_name = os.path.join(self.image_folder_path, self.poke_frame.iloc[idx, 0])
    image = Image.open(image_name).convert('RGB')

    poke = self.poke_frame.iloc[idx, 1:4] # TODO: update so matches with new csv format
    poke = np.array([poke], dtype='float32')

    sample = {'image': image, 'poke': poke}

    if self.transforms:
      for k in sample.keys():
        sample[k] = self.transforms(sample[k])

    return sample


def get_episodes():
    """**Load .csv file as Python object**"""

    data_file = open('data/test.csv', 'r')
    reader = csv.reader(data_file)
    header = next(reader)
    data = {}

    print("Headers:           ",header) # ['image_file_name', '∆x', '∆y', '∆z', 'episode']
    for h in header:
      data[h] = []

    for row in reader:
      for h, v in zip(header, row):
        data[h].append(v)

    print("Number of episodes:", len(np.unique(data.get('episode'))))
    print("Number of images:  ", len(data.get('image_file_name')))
    print("Images per episode:", np.unique(data.get('episode'), return_counts=True)[1])
    assert min(np.unique(data.get('episode'), return_counts=True)[1]) > 0, "Episode with fewer images than expected." # TODO: change to 10

    # Train, Valid, Test (70%, 20%, 10%)
    train_frac = 0.7
    valid_frac = 0.2
    test_frac = 0.1

    episode_indeces = np.unique(data.get('episode'))
    num_episodes = len(episode_indeces)
    random.shuffle(episode_indeces)

    train_bound = int(num_episodes * train_frac)
    valid_bound = int(train_bound + num_episodes * valid_frac)

    train_episodes = episode_indeces[0 : train_bound]
    valid_episodes = episode_indeces[train_bound : valid_bound]
    test_episodes = episode_indeces[valid_bound : ]

    print("Episodes:")
    print(("Train: %d | ~%.2f") % (len(train_episodes), len(train_episodes)/num_episodes))
    print(("Valid: %d | ~%.2f") % (len(valid_episodes), len(valid_episodes)/num_episodes))
    print(("Test : %d | ~%.2f") % (len(test_episodes), len(test_episodes)/num_episodes))

    return train_episodes, valid_episodes, test_episodes

#data = torchvision.datasets.CIFAR10('~/data', train=True, download=True)
#mu = data.data.mean(axis=(0, 1, 2)) # (N, H, W, 3) -> 3
#std = data.data.std(axis=(0, 1, 2)) # (N, H, W, 3) -> 3
#print(data.data.shape)

def PokeData(episodes, trnsfrms=None):
  tfms_norm = torchvision.transforms.Compose([
      transforms.ToTensor()
      # ToTensor already maps 0-255 to 0-1, so divide mu and std by 255 below
      #transforms.Normalize(mu / 255, std /255),
  ])
  tf = transforms.Compose([trnsfrms, tfms_norm]) if trnsfrms is not None else tfms_norm
  return PokeDataset('data/test.csv', 'data/images/', episodes, tf)


def get_data_loaders(train_episodes, valid_episodes, test_episodes):
    loader_kwargs = {'batch_size': 16, 'num_workers': 2}
    train_dataset = PokeData(train_episodes)
    valid_dataset = PokeData(valid_episodes)
    test_dataset = PokeData(test_episodes)

    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **loader_kwargs)

    return train_loader, valid_loader, test_loader

#def get_loader_train(augmentations=lambda x: x):
#    train_dataset = PokeData(train_episodes, transforms.Compose([augmentations]))
#    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs, shuffle=True)
#    return train_loader

def correct_poke(est, gt):
  normalized_est = est / np.linalg.norm(est, axis=1).reshape(-1,1)
  normalized_gt = gt / np.linalg.norm(gt, axis=1).reshape(-1,1)
  return np.sum(0.1 > np.linalg.norm(normalized_gt - normalized_est, axis=1))

def one_epoch(model, data_loader, opt=None):
    device = next(model.parameters()).device
    train = False if opt is None else True
    model.train() if train else model.eval()
    losses, correct, total = [], 0, 0
    for data in data_loader:
        x = data['image']
        y = data['poke'].squeeze()
        #print("x.shape", x.shape)
        #print("y.shape", y.shape)
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
        loss = F.mse_loss(logits, y)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append(loss.item())
        total += len(x)
        correct += correct_poke(y.cpu().detach().numpy(), logits.cpu().detach().numpy()) #(torch.argmax(logits, dim=1) == y).sum().item()
    return np.mean(losses), correct / total


def train(model, loader_train, loader_valid, lr=1e-3, max_epochs=30, weight_decay=0., patience=3):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_accuracy = 0
    best_valid_accuracy_epoch = 0

    t = tqdm(range(max_epochs))
    for epoch in t:
        train_loss, train_acc = one_epoch(model, loader_train, opt)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        valid_loss, valid_acc = one_epoch(model, loader_valid)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        t.set_description(f'train_acc: {train_acc:.2f}, valid_acc: {valid_acc:.2f}')

        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc
            best_valid_accuracy_epoch = epoch

        if epoch > best_valid_accuracy_epoch + patience:
            break
    t.set_description(f'best valid acc: {best_valid_accuracy:.2f}')

    return train_losses, train_accuracies, valid_losses, valid_accuracies

def plot_history(train_losses, train_accuracies, valid_losses, valid_accuracies):
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    p = plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.autoscale()
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    p = plt.plot(train_accuracies, label='train')
    plt.plot(valid_accuracies, label='valid')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def get_model():
    model = torchvision.models.resnet18(pretrained=True)
    #print(model)
    # See that the head after the conv-layers (in the bottom) is one linear layer, from 512 features to 1k-class logits.
    # We want to replace it with a new head to 10-class logits:
    model.fc = nn.Linear(512, 3)
    # Also, the model has been trained on images with a resolution of 224. Let's upscale our cifar10 images:

    model = nn.Sequential(
        nn.UpsamplingBilinear2d((224,224)),
        model,
    )
    model = model.cuda()

    return model
