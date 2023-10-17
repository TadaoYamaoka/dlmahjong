import torch
from torch import nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out + x)

class PolicyHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 出力
        # 打牌 34+3(赤牌)
        # 自摸切り 1
        # チー 3(パターン) x 2(赤牌有無)
        # ポン 1 x 2(赤牌有無)
        # カン 1
        # 鳴かない 1
        # 明槓 34
        # 加槓 34
        # 立直 1
        # 自摸和了 1
        # ロン 1

        self.fc1 = nn.Linear(channels * 9 * 4, 256)
        self.fc2 = nn.Linear(256, 118)

        # 補助タスク1
        # 役 66(場風・自風はそれぞれ1、翻牌は牌別、ドラと裏ドラはそれぞれ10までカウント)
        self.fc1_aux1 = nn.Linear(channels * 9 * 4, 256)
        self.fc2_aux1 = nn.Linear(256, 66)

        # 補助タスク2
        # 和了プレイヤー 4+流局1
        self.fc1_aux2 = nn.Linear(channels * 9 * 4, 256)
        self.fc2_aux2 = nn.Linear(256, 5)

        # 補助タスク3
        # 他家の待ち牌 34 x 3
        self.fc1_aux3 = nn.Linear(channels * 9 * 4, 256)
        self.fc2_aux3 = nn.Linear(256, 102)

    def forward(self, x):
        p = self.fc1(x)
        p = F.relu(p)
        p = self.fc2(p)

        aux1 = self.fc1_aux1(x)
        aux1 = F.relu(aux1)
        aux1 = self.fc2_aux1(aux1)

        aux2 = self.fc1_aux2(x)
        aux2 = F.relu(aux2)
        aux2 = self.fc2_aux2(aux2)

        aux3 = self.fc1_aux3(x)
        aux3 = F.relu(aux3)
        aux3 = self.fc2_aux3(aux3)

        return p, aux1, aux2, aux3

class ValueHead(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        # 価値入力チャンネル
        # 他家の手牌 7(牌種4+赤牌3) x 3(プレイヤー)
        # 他家の聴牌 1 x 3(プレイヤー)
        # 残り牌 7(牌種4+赤牌3)
        # 裏ドラ 7(牌種4+赤牌3)
        self.conv1 = nn.Conv2d(channels + 38, channels, kernel_size=3, padding=1)
        # Resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])
        # fcl
        self.fc1 = nn.Linear(channels * 9 * 4, 256)
        # 出力 報酬
        self.fc2 = nn.Linear(256, 1)

        # 補助タスク 点数(4プレイヤー)
        self.fc1_aux = nn.Linear(channels * 9 * 4, 256)
        self.fc2_aux = nn.Linear(256, 4)

    def forward(self, x1, x2):
        x = self.conv1(torch.cat((x1, x2), dim=1))
        x = F.relu(x)
        x = self.blocks(x)
        x = x.flatten(1)

        v = self.fc1(x)
        v = F.relu(v)
        v = self.fc2(v)

        aux = self.fc1_aux(x)
        aux = F.relu(aux)
        aux = self.fc2_aux(aux)

        return v, aux

class PolicyValueNet(nn.Module):
    def __init__(self, channels, blocks, value_blocks):
        super().__init__()
        # 方策の入力チャンネル
        # 状態 5(打牌、副露x3他家、副露)
        # 手牌 7(牌種4+赤牌3)
        # 副露 (チー3(牌種) + ポン・カン4(牌種) + 暗槓4(牌種) + 赤牌3(牌種)) x 4(プレイヤー)
        # 自摸牌 7(牌種4+赤牌3)
        # 他家打牌 7(牌種4+赤牌3)
        # 聴牌 1
        # 立直 4
        # 河牌 (7(牌種4+赤牌3) + 立直後捨て牌(牌種4)) x 4(プレイヤー)
        # 他家の直前の捨て牌 4
        # ドラ 7(牌種4+赤牌3)
        # 自風 4
        # 場風 4
        # 残り牌数 1
        # エージェント番号 4
        self.conv1 = nn.Conv2d(155, channels, kernel_size=3, padding=1)
        # Resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])
        # Policy head
        self.policy_head = PolicyHead(channels)
        # Value head
        self.value_head = ValueHead(channels, value_blocks)

    def extract_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.blocks(x)
        return x

    def forward(self, x1, x2):
        x1 = self.extract_features(x1)

        # Policy head
        p = self.policy_head(x1.flatten(1))

        # Value head
        v = self.value_head(x1, x2)

        return p, v
