import torch
from dlmahjong.model import PolicyValueNet
from cmajiang import (
    Game,
    public_features,
    private_features,
    N_CHANNELS_PUBLIC,
    N_CHANNELS_PRIVATE,
)
import numpy as np


def test_forward():
    game = Game()
    game.kaiju()
    game.qipai()
    game.zimo()

    features1 = torch.zeros((1, N_CHANNELS_PUBLIC + 4, 9, 4), dtype=torch.float32)
    features2 = torch.zeros((1, N_CHANNELS_PRIVATE, 9, 4), dtype=torch.float32)
    public_features(game, 0, features1.numpy())
    private_features(game, 0, features2.numpy())

    device = torch.device("cuda")
    model = PolicyValueNet(channels=128, blocks=10, value_blocks=5)
    model.to(device)
    (p, p_aux1, p_aux2, p_aux3), (v, v_aux) = model(
        features1.to(device), features2.to(device)
    )

    assert p.shape == (1, 118)
    assert p_aux1.shape == (1, 66)
    assert p_aux2.shape == (1, 5)
    assert p_aux3.shape == (1, 102)
    assert v.shape == (1, 1)
    assert v_aux.shape == (1, 4)
