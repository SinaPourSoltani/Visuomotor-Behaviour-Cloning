import os
from zipfile import ZipFile
import numpy as np
import pandas as pd
import csv
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utilities import Geometry as geo
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, transforms

import quaternion

def load_data(data_link):
    if not os.path.exists('data'):
      os.system("wget " + data_link)
      file_name = data_link.split('/')[-1]
      with ZipFile(file_name, 'r') as z:
        z.extractall("./")

class PokeDataset(Dataset):
  def __init__(self, csv_file_path, image_folder_path, episode_indeces, transforms=None, is_stereo=False, std_noise_poke_vec=None):

    self.image_folder_path = image_folder_path
    self.transforms = transforms
    self.is_stereo = is_stereo
    self.poke_frame = pd.read_csv(csv_file_path)
    self.poke_frame = self.poke_frame.loc[self.poke_frame['episode'].isin(list(map(int,episode_indeces)))]
    self.stepsize = 0.01
    self.std_noise_poke_vec = std_noise_poke_vec
    self.transforms = transforms

  def __len__(self):
    return len(self.poke_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    image = None
    image_l = None
    image_r = None

    if not self.is_stereo:
        image_name = os.path.join(self.image_folder_path, self.poke_frame.iloc[idx, 0])
        image = Image.open(image_name).convert('RGB')
    else:
        image_name_l = os.path.join(self.image_folder_path, self.poke_frame.iloc[idx, 0])
        image_l = Image.open(image_name_l).convert('RGB')
        image_name_r = os.path.join(self.image_folder_path, self.poke_frame.iloc[idx, 1])
        image_r = Image.open(image_name_r).convert('RGB')

    stereo_offset = 1 if self.is_stereo else 0
    poke = self.poke_frame.iloc[idx, 1+stereo_offset : 4+stereo_offset] # TODO: update so matches with new csv format

    poke = np.array([poke], dtype='float32')
    poke = poke / self.stepsize
    if self.std_noise_poke_vec is not None:
          poke = add_noise_poke_vector(poke)

    sample = {'image': self.transforms(image)} if not self.is_stereo \
        else {'image_l': self.transforms(image_l), 'image_r': self.transforms(image_r)}
    sample['poke'] = torch.tensor(poke)

    return sample

def add_noise_poke_vector(vec, std_dev_deg=1):
  theta_phi = np.radians(np.random.normal(0, std_dev_deg, size=(1, 2)))
  q = quaternion.from_spherical_coords(theta_phi)
  vec_rot = quaternion.rotate_vectors(q, vec)
  return vec_rot[0]

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
    print("Number of images:  ", len(data.get(list(data.keys())[0])))
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

def PokeData(episodes, trnsfrms=None, is_stereo=False, std_noise_poke_vec=None):
  tfms_norm = torchvision.transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  tf = transforms.Compose([trnsfrms, tfms_norm]) if trnsfrms is not None else tfms_norm
  return PokeDataset('data/test.csv', 'data/images/', episodes, tf, is_stereo, std_noise_poke_vec=std_noise_poke_vec)


def get_data_loaders(train_episodes, valid_episodes, test_episodes, transforms=None, is_stereo=False, std_noise_poke_vec=None):
    loader_kwargs = {'batch_size': 16, 'num_workers': 2}
    train_dataset = PokeData(train_episodes, transforms, is_stereo, std_noise_poke_vec=std_noise_poke_vec)
    valid_dataset = PokeData(valid_episodes, transforms, is_stereo)
    test_dataset = PokeData(test_episodes, transforms, is_stereo)

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

def one_epoch(model, data_loader, opt=None, is_stereo=False):
    device = next(model.parameters()).device
    train = False if opt is None else True
    model.train() if train else model.eval()
    losses, total_deviation, total = [], 0, 0
    for data in data_loader:
        x = x1 = x2 = None
        if is_stereo:
            x1 = data['image_l'].to(device)
            x2 = data['image_r'].to(device)
        else:
            x = data['image'].to(device)
        y = data['poke'].squeeze().to(device)

        with torch.set_grad_enabled(train):
            if not is_stereo:
              logits = model(x)
            else:
              logits = model(x1, x2)


        loss = F.mse_loss(logits, y)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append(loss.item())
        total += len(x) if not is_stereo else len(x1)
        for vector_pred, vector_gt in zip(logits, y):
          total_deviation += geo.angle_between_vectors(vector_pred.cpu().detach().numpy(), vector_gt.cpu().detach().numpy())
        #correct += correct_poke(y.cpu().detach().numpy(), logits.cpu().detach().numpy()) #(torch.argmax(logits, dim=1) == y).sum().item()
    return np.mean(losses), total_deviation / total


def train(model, loader_train, loader_valid, lr=1e-3, max_epochs=30, weight_decay=0., patience=3, is_stereo=False, model_tag=""):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_accuracy = 0
    best_valid_accuracy_epoch = 0

    summary = {}

    t = tqdm(range(max_epochs))
    for epoch in t:
        train_loss, train_acc = one_epoch(model, loader_train, opt, is_stereo=is_stereo)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        valid_loss, valid_acc = one_epoch(model, loader_valid, is_stereo=is_stereo)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        t.set_description(f'train_loss: {train_loss:.10f}, valid_loss: {valid_loss:.10f}')

        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc
            best_valid_accuracy_epoch = epoch


        if (epoch + 1) % 5 == 0:
            filename = "ResNet18_epoch" + str(epoch + 1) + "_" + model_tag + ".pth"
            torch.save(model.state_dict(), filename)

            #summary["Model name"] = filename;
            summary["Epochs"] = len(train_losses)
            summary["Train losses"] = train_losses
            summary["Train deviation"] = train_accuracies
            summary["Valid losses"] = valid_losses
            summary["Valid deviation"] = valid_accuracies
            summary["Is stereo"] = is_stereo
            summary["Learning rate"] = lr

            summary_file_name = "summary_" + model_tag
            with open(summary_file_name, 'w') as outfile:
                json.dump(summary, outfile)

        if epoch > best_valid_accuracy_epoch + patience and patience >= 0:
            break
    #t.set_description(f'best valid acc: {best_valid_accuracy:.2f}')

    return train_losses, train_accuracies, valid_losses, valid_accuracies

def plot_history(train_losses, train_accuracies, valid_losses, valid_accuracies):
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss - MSE')
    p = plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.autoscale()
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('angular deviation [rad]')
    p = plt.plot(train_accuracies, label='train')
    plt.plot(valid_accuracies, label='valid')
    plt.autoscale()
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def get_model(complex_mlp=False, is_stereo=False):
    model = PokeNet(complex_mlp=complex_mlp, is_stereo=is_stereo)
    try:
      model = model.cuda()
    except:
      model = model.cpu()
    return model

def freeze_backbone(model, is_stereo=False):

    backbone = list(model.named_children())[0][1]
    for idx,  (name, layer) in enumerate(backbone.named_children()):
        layer.requires_grad = False

    if is_stereo:
        backbone2 = list(model.named_children())[1][1]
        for idx,  (name, layer) in enumerate(backbone2.named_children()):
          layer.requires_grad = False

    return model

def unfreeze_backbone(model, is_stereo=False):
    backbone = list(model.named_children())[0][1]

    for idx,  (name, layer) in enumerate(backbone.named_children()):
      if idx > 5: #Only unfreeze above Residual layer 2 and above.
        layer.requires_grad = True

    if is_stereo:
      backbone2 = list(model.named_children())[1][1]
      for idx,  (name, layer) in enumerate(backbone2.named_children()):
        if idx > 5: #Only unfreeze above Residual layer 2 and above.
          layer.requires_grad = True

    return model

class PokeNet(nn.Module):
    def __init__(self, complex_mlp=False, is_stereo=False, p_dropout=0.):
        super().__init__()
        self.is_stereo = is_stereo
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()

        if is_stereo:
            self.backbone_2 = torchvision.models.resnet18(pretrained=True)
            self.backbone_2.fc = nn.Identity()

        if complex_mlp:
          self.head = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(1024, 256) if self.is_stereo else nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p_dropout*0.5),
            nn.Linear(256, 3)
            )
        else:
          self.head = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(1024, 3) if self.is_stereo else nn.Linear(512, 3)
          )

        with torch.no_grad():
          self.head[1].weight.fill_(0)
          self.head[1].bias.fill_(0)
          if complex_mlp:
            self.head[4].weight.fill_(0)
            self.head[4].bias.fill_(0)

    def forward(self, x1, x2=None):
        if self.is_stereo:
            x1 = self.backbone(x1)
            x2 = self.backbone_2(x2)
            x = torch.cat((x1, x2), dim=1)
        else:
            x = self.backbone(x1)

        poke = self.head(x)
        unit_poke = poke / (torch.linalg.norm(poke) + 0.00001)
        return unit_poke
