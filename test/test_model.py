import torch
from dlmahjong.model import PolicyValueNet
from cmajiang import (
    Game,
    game_public_features,
    game_private_features,
    N_CHANNELS_PUBLIC,
    N_CHANNELS_PRIVATE,
)
import numpy as np


def test_forward():
    game = Game()
    game.kaiju()
    game.qipai()
    game.zimo()

    public_featuers = torch.zeros((1, N_CHANNELS_PUBLIC + 4, 9, 4), dtype=torch.float32)
    private_featuers = torch.zeros((1, N_CHANNELS_PRIVATE, 9, 4), dtype=torch.float32)
    game_public_features(game, 0, public_featuers.numpy())
    game_private_features(game, 0, private_featuers.numpy())

    device = torch.device("cuda")
    model = PolicyValueNet(channels=128, blocks=10, value_blocks=5)
    model.to(device)
    (p, p_aux1, p_aux2, p_aux3), (v, v_aux) = model(
        public_featuers.to(device), private_featuers.to(device)
    )

    assert p.shape == (1, 118)
    assert p_aux1.shape == (1, 66)
    assert p_aux2.shape == (1, 5)
    assert p_aux3.shape == (1, 102)
    assert v.shape == (1, 1)
    assert v_aux.shape == (1, 4)
