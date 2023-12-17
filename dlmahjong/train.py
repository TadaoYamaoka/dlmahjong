import argparse
from typing import NamedTuple
import glob
import os
import zlib
import time
import numpy as np

import torch
from torch.utils.data import Dataset

from cmajiang import N_CHANNELS_PUBLIC, N_CHANNELS_PRIVATE, N_ACTIONS

parser = argparse.ArgumentParser()
parser.add_argument("basedir")
args = parser.parse_args()

PublicFeatures = np.dtype((np.float32, (N_CHANNELS_PUBLIC + 4, 9, 4)))
PrivateFeatures = np.dtype((np.float32, (N_CHANNELS_PRIVATE, 9, 4)))
Policy = np.dtype((np.float32, N_ACTIONS))
Hupai = np.dtype((np.float32, 54))
HulePlayer = np.dtype((np.float32, 5))
TajiaTingpai = np.dtype((np.float32, (3, 34)))
Fenpei = np.dtype((np.float32, 4))
StepData = np.dtype(
    [
        ("public_features", PublicFeatures),
        ("private_features", PrivateFeatures),
        ("action", np.int64),
        ("value", np.float32),
        ("log_probs", Policy),
        ("advantage", np.float32),
        ("hupai", Hupai),
        ("hule_player", HulePlayer),
        ("tajia_tingpai", TajiaTingpai),
        ("fenpei", Fenpei),
    ]
)


class RolloutDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.rollout_data = np.empty(0, StepData)

    def load(self, path):
        with open(path, "rb") as f:
            data = zlib.decompress(f.read())
        tmp = np.frombuffer(data, StepData)
        self.rollout_data = np.concatenate((self.rollout_data, tmp))

    def __len__(self):
        return len(self.rollout_data)

    def __getitem__(self, idx):
        data = self.rollout_data[idx]
        
        return data["public_features"], data["private_features"], data["action"], data["value"], data["log_probs"], data["advantage"], data["advantage"] + data["value"]


max_dir_num = -1
for dir_name in os.listdir(args.basedir):
    if dir_name.isdigit():
        dir_num = int(dir_name)
        if dir_num > max_dir_num:
            max_dir_num = dir_num
            path = os.path.join(args.basedir, dir_name)
if max_dir_num < 0:
    max_dir_num = 0
    path = os.path.join(args.basedir, str(max_dir_num))
    os.makedirs(path)

# stopファイルがある場合、次のディレクトリへ
if os.path.exists(os.path.join(path, "stop")):
    max_dir_num += 1
    path = os.path.join(args.basedir, str(max_dir_num))
    os.makedirs(path)

processed = set()
dataset = RolloutDataset()
while True:
    for data_path in sorted(glob.glob(os.path.join(path, "*.dat"))):
        if data_path in processed:
            continue

        dataset.load(data_path)
        processed.add(data_path)

    time.sleep(1)
