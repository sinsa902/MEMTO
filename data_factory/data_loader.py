from copy import deepcopy
from pathlib import Path

import torch
import os
import random
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.utils.data import DataLoader
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

import lib as lb


class ConvertToTorchdataset(Dataset):
    def __init__(self, data):
        self.X_num = data["X_num"]
        self.X_cat = data["X_cat"]
        self.Y = data["Y"]

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.Y[idx]


class AdultLoader:
    def __init__(self, data_path):
        D = lb.Dataset.from_dir(data_path)

        X = D.build_X(
            normalization="quantile",
            num_nan_policy="mean",
            cat_nan_policy="new",
            cat_policy="indices",
            cat_min_frequency=0.0,
            seed=0,
        )
        self.X_num, self.X_cat = X
        self.Y = D.build_y(None)

    def add_dictionary_trainvaltest(self):
        keys = ["train", "val", "test"]

        transformed_data = {key: {} for key in keys}

        for key in keys:
            transformed_data[key]["X_num"] = self.X_num.get(key, None)
            transformed_data[key]["X_cat"] = self.X_cat.get(key, None)
            transformed_data[key]["Y"] = self.Y.get(key, None)

        self.train = transformed_data["train"]
        self.val = transformed_data["val"]
        self.test = transformed_data["test"]


def get_loader_segment(
    data_path,
    batch_size,
    dataset="KDD",
):
    """
    model : 'train' or 'test'
    """
    if dataset == "adult":
        dataset = AdultLoader(data_path)

    dataset.add_dictionary_trainvaltest()

    train_dataset = ConvertToTorchdataset(dataset.train)
    val_dataset = ConvertToTorchdataset(dataset.val)
    test_dataset = ConvertToTorchdataset(dataset.test)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    k_use_len = int(len(train_dataset) * 0.1)
    indices = torch.arange(len(train_dataset))
    k_sub_indices = indices[:k_use_len]
    k_subset = Subset(train_dataset, k_sub_indices)
    k_loader = DataLoader(
        dataset=k_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return dataset, train_loader, val_loader, k_loader, test_loader
