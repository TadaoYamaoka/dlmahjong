import argparse
import torch

from cmajiang import N_CHANNELS_PUBLIC
from dlmahjong.model import PolicyValueNet, PolicyNet

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("onnx")
args = parser.parse_args()

pv_model = PolicyValueNet(channels=128, blocks=10, value_blocks=5)
model = PolicyNet(pv_model)

model.eval()

x = torch.randn((1, N_CHANNELS_PUBLIC + 4, 9, 4), dtype=torch.float32)

torch.onnx.export(
    model,
    x,
    args.onnx,
    verbose=True,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"},},
)
