import argparse
from typing import NamedTuple
import glob
import os

import torch

from cmajiang import Paipu, PaipuReplay, Status

parser = argparse.ArgumentParser()
parser.add_argument("basedir")
args = parser.parse_args()

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
while True:
    for paipu_path in sorted(glob.glob(os.path.join(path, "*.paipu"))):
        if paipu_path in processed:
            continue

        with open(paipu_path, "rb") as f:
            for line in f.readlines():
                paipu = Paipu(line)
                replay = PaipuReplay(paipu)
                while replay.status != Status.JIEJI:
                    game = replay.game
                    
                    replay.next()
