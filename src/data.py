import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms.transforms import RandomAffine, RandomPerspective


HEIGHT = None
WIDTH = None


class Dataset(data.Dataset):
    def __init__(self, csv_path, transform):
        self.transform = transform
        self.length = None
        data = pd.read_csv(csv_path)
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self.length


def get_train_loader(csv_path, batch_size, num_workers):
    """Returns data set and data loader for training."""
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.3),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.12))])
    dataset = Dataset(csv_path, transform)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=True)
    return dataset, loader


def get_test_loader(csv_path, batch_size, num_workers):
    """Returns data set and data loader for evaluation."""
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor()])
    dataset = Dataset(csv_path, transform)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False)
    return dataset, loader
