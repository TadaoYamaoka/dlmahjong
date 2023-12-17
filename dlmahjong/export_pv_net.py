import argparse
import torch

from cmajiang import N_CHANNELS_PUBLIC, N_CHANNELS_PRIVATE
from dlmahjong.model import PolicyValueNetWithAux, PolicyValueNet

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("onnx")
args = parser.parse_args()

pv_model = PolicyValueNetWithAux(channels=128, blocks=10, value_blocks=5)
model = PolicyValueNet(pv_model)

model.eval()

x1 = torch.randn((1, N_CHANNELS_PUBLIC + 4, 9, 4), dtype=torch.float32)
x2 = torch.randn((1, N_CHANNELS_PRIVATE, 9, 4), dtype=torch.float32)

torch.onnx.export(
    model,
    (x1, x2),
    args.onnx,
    verbose=True,
    do_constant_folding=True,
    input_names=["input1", "input2"],
    output_names=["output1", "output2"],
    dynamic_axes={
        "input1": {0: "batch_size"},
        "input2": {0: "batch_size"},
        "output1": {0: "batch_size"},
        "output2": {0: "batch_size"},
    },
)
