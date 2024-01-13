import argparse
from typing import NamedTuple
import glob
import os
import zlib
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from cmajiang import N_CHANNELS_PUBLIC, N_CHANNELS_PRIVATE, N_ACTIONS
from dlmahjong.model import PolicyValueNetWithAux


parser = argparse.ArgumentParser()
parser.add_argument("basedir")
parser.add_argument("--n_rollout_steps", type=int, default=1048576)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
args = parser.parse_args()


# PPO Parameter
clip_range = 0.2
normalize_advantage = True
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 0.5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        ("action", np.dtype((np.int64, (1, )))),
        ("value", np.float32),
        ("logits", Policy),
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
        
        return data["public_features"], data["private_features"], data["action"], data["logits"], data["advantage"], data["advantage"] + data["value"], data["hupai"]


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

    # 一定数データが溜まったら学習開始
    if len(dataset) >= args.n_rollout_steps:
        break

    time.sleep(3)


rollout_buffer = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

model = PolicyValueNetWithAux(channels=128, blocks=10, value_blocks=5)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

for epoch in range(args.n_epochs):
    for public_features, private_features, actions, logits, advantages, returns, hupai in rollout_buffer:
        public_features = public_features.to(device)
        private_features = private_features.to(device)
        actions = actions.to(device)
        logits = logits.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        hupai = hupai.to(device)

        with torch.inference_mode():
            old_log_prob = model.log_prob(actions, logits)

        values, log_prob, entropy, p_aux1, p_aux2, p_aux3, v_aux = model.evaluate_actions_with_aux(public_features, private_features, actions)

        values = values.flatten()

        # Normalize advantage
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(log_prob - old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(returns, values)

        # Entropy loss favor exploration
        entropy_loss = -torch.mean(entropy)

        # 補助タスク1 役
        p_aux1_loss = bce_with_logits_loss(p_aux1, hupai)

        loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss + p_aux1_loss

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
