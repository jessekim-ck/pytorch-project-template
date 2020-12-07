import os

import numpy as np
import pandas as pd

import torch
from torch.utils import data
from torchvision import transforms


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
        transforms.Resize((None, None)),  # TODO
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ColorJitter(),
        transforms.RandomErasing()])
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
        transforms.Resize((None, None)),  # TODO
        transforms.ToTensor()])
    dataset = Dataset(csv_path, transform)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False)
    return dataset, loader
