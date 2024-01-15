import os
import torch

from cmajiang import N_CHANNELS_PUBLIC, N_CHANNELS_PRIVATE
from dlmahjong.model import PolicyValueNetWithAux, PolicyValueNet
from dlmahjong.checkpoint import save_checkpoint

def export_onnx(model, path, device="cpu", verbose=False):
    onnx_path = os.path.join(path, "pv_net.onnx")
    pv_model = PolicyValueNet(model)

    pv_model.eval()

    x1 = torch.randn((1, N_CHANNELS_PUBLIC + 4, 9, 4), dtype=torch.float32).to(device)
    x2 = torch.randn((1, N_CHANNELS_PRIVATE, 9, 4), dtype=torch.float32).to(device)

    torch.onnx.export(
        pv_model,
        (x1, x2),
        onnx_path,
        verbose=verbose,
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("basedir")
    args = parser.parse_args()

    path = os.path.join(args.basedir, "0")
    os.makedirs(path, exist_ok=True)

    model = PolicyValueNetWithAux()

    save_checkpoint(model, None, 0, path)
    export_onnx(model, path, verbose=True)

    # startファイル作成
    with open(os.path.join(path, "start"), "w") as f:
        pass
