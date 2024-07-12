from copy import deepcopy
from pathlib import Path

import torch
import os
import random
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

from lib import normalize


# import lib


class AdultLoader(Dataset):
    def __init__(self, data_path, mode="train"):
        self.mode = mode
        dir_ = Path(data_path)

        def load_numpy(data_type):
            return {
                x: np.load(dir_ / f"{data_type}_{x}.npy")  # type: ignore[code]
                for x in ["train", "val", "test"]
            }

        if dir_.joinpath("N_train.npy").exists():
            self.n_data = load_numpy("N")
        if dir_.joinpath("C_train.npy").exists():
            self.c_data = load_numpy("C")
        if dir_.joinpath("y_train.npy").exists():
            self.y_data = load_numpy("y")

        scale_norm = StandardScaler()
        n_data = {k: np.nan_to_num(v) for k, v in self.n_data.items()}
        for k, v in n_data.items():
            scale_norm.fit(v)
            n_data[k] = scale_norm.transform(v)

        N = deepcopy(self.n_data)

        num_nan_masks = {k: np.isnan(v) for k, v in N.items()}
        if any(x.any() for x in num_nan_masks.values()):  # type: ignore[code]
            num_new_values = np.nanmean(self.n_data["train"], axis=0)
            for k, v in N.items():
                num_nan_indices = np.where(num_nan_masks[k])
                v[num_nan_indices] = np.take(num_new_values, num_nan_indices[1])
            N = normalize(N, normalization, seed)
        print("numeric shape:", self.n_data.shape)
        print("categorical shape:", self.c_data.shape)
        print("label shape:", self.y_data.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        mode : "train" or "test"
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )


class SMAPSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        scale_norm = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        scale_norm.fit(data)
        data = scale_norm.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = scale_norm.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return (np.float32(self.train[index : index + self.win_size])), (
                np.float32(self.test_labels[0 : self.win_size])
            )
        elif self.mode == "test":
            return (np.float32(self.test[index : index + self.win_size])), (
                np.float32(self.test_labels[index : index + self.win_size])
            )
        else:
            return (np.float32(self.train[index : index + self.win_size])), (
                np.float32(self.test_labels[0 : self.win_size])
            )


def get_loader_segment(
    data_path,
    batch_size,
    win_size=100,
    step=100,
    mode="train",
    dataset="KDD",
    val_ratio=0.2,
):
    """
    model : 'train' or 'test'
    """
    if dataset == "SMD":
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == "MSL":
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif dataset == "SMAP":
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif dataset == "PSM":
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif dataset == "SWaT":
        dataset = SWaTSegLoader(data_path, win_size, step, mode)
    elif dataset == "adult":
        dataset = AdultLoader(data_path, mode)

    shuffle = False
    if mode == "train":
        shuffle = True

        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)

        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len :]]
        )
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index : val_start_index + val_use_len]
        val_subset = Subset(dataset, val_sub_indices)

        train_loader = DataLoader(
            dataset=train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )
        val_loader = DataLoader(
            dataset=val_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )

        k_use_len = int(train_use_len * 0.1)
        k_sub_indices = indices[:k_use_len]
        k_subset = Subset(dataset, k_sub_indices)
        k_loader = DataLoader(
            dataset=k_subset, batch_size=batch_size, shuffle=shuffle, num_workers=0
        )

        return train_loader, val_loader, k_loader

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )

    return data_loader, data_loader
