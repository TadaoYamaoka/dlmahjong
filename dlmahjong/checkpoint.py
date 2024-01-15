import torch
import os

def save_checkpoint(model, optimizer, state, path):
    # チェックポイント保存
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        **state
    }
    torch.save(checkpoint, os.path.join(path, "checkpoint.pth"))

def load_checkpoint(model, optimizer, state, path):
    checkpoint = torch.load(os.path.join(path, "checkpoint.pth"))

    model_state_dict = checkpoint.pop("model")
    model.load_state_dict(model_state_dict)

    optimizer_state_dict = checkpoint.pop("optimizer")
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    state.update(checkpoint)
