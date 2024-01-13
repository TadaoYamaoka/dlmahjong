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
        # 鳴かない・ロンしない 1
        # 打牌 34+3(赤牌)
        # 自摸切り 1
        # 立直 打牌・自摸切り 34+3+1
        # チー 3(パターン) x 2(赤牌有無)
        # ポン 1 x 2(赤牌有無)
        # カン 1
        # 暗槓・加槓 34
        # 和了 1

        self.fc1 = nn.Linear(channels * 9 * 4, 256)
        self.fc2 = nn.Linear(256, 121)

        # 補助タスク1
        # 役 54(場風・自風はそれぞれ1、翻牌は牌別、ドラと裏ドラはそれぞれ4までカウント)
        self.fc1_aux1 = nn.Linear(channels * 9 * 4, 256)
        self.fc2_aux1 = nn.Linear(256, 54)

        # 補助タスク2
        # 和了プレイヤー 4+流局1
        self.fc1_aux2 = nn.Linear(channels * 9 * 4, 256)
        self.fc2_aux2 = nn.Linear(256, 5)

        # 補助タスク3
        # 他家の待ち牌 34 x 3
        self.fc1_aux3 = nn.Linear(channels * 9 * 4, 256)
        self.fc2_aux3 = nn.Linear(256, 102)

    def forward_policy(self, x):
        p = self.fc1(x)
        p = F.relu(p)
        p = self.fc2(p)
        return p

    def forward_aux1(self, x):
        aux1 = self.fc1_aux1(x)
        aux1 = F.relu(aux1)
        aux1 = self.fc2_aux1(aux1)
        return aux1

    def forward_aux2(self, x):
        aux2 = self.fc1_aux2(x)
        aux2 = F.relu(aux2)
        aux2 = self.fc2_aux2(aux2)
        return aux2

    def forward_aux3(self, x):
        aux3 = self.fc1_aux3(x)
        aux3 = F.relu(aux3)
        aux3 = self.fc2_aux3(aux3)
        return aux3

    def forward(self, x):
        p = self.forward_policy(x)
        p_aux1 = self.forward_aux1(x)
        p_aux2 = self.forward_aux2(x)
        p_aux3 = self.forward_aux3(x)

        return p, p_aux1, p_aux2, p_aux3

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

    def forward_value(self, x1, x2):
        x = self.conv1(torch.cat((x1, x2), dim=1))
        x = F.relu(x)
        x = self.blocks(x)
        x = x.flatten(1)

        v = self.fc1(x)
        v = F.relu(v)
        v = self.fc2(v)

        return v

    def forward(self, x1, x2):
        x = self.conv1(torch.cat((x1, x2), dim=1))
        x = F.relu(x)
        x = self.blocks(x)
        x = x.flatten(1)

        v = self.fc1(x)
        v = F.relu(v)
        v = self.fc2(v)

        v_aux = self.fc1_aux(x)
        v_aux = F.relu(v_aux)
        v_aux = self.fc2_aux(v_aux)

        return v, v_aux

class PolicyValueNetWithAux(nn.Module):
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
        x = self.extract_features(x1)

        # Policy head
        p, p_aux1, p_aux2, p_aux3 = self.policy_head(x.flatten(1))

        # Value head
        v, v_aux = self.value_head(x, x2)

        return p, p_aux1, p_aux2, p_aux3, v, v_aux

    def forward_policy(self, x):
        x = self.extract_features(x)
        p = self.policy_head.forward_policy(x.flatten(1))
        return p

    def forward_policy_value(self, x1, x2):
        x = self.extract_features(x1)
        p = self.policy_head.forward_policy(x.flatten(1))
        v = self.value_head.forward_value(x, x2)
        return p, v

    @staticmethod
    def log_prob(value, logits):
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        log_prob = log_pmf.gather(-1, value).squeeze(-1)
        return log_prob

    @staticmethod
    def entropy(logits):
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        probs = F.softmax(logits, dim=-1)
        p_log_p = logits * probs
        return -p_log_p.sum(-1)

    def evaluate_actions(self, public_features, private_features, actions):
        logits, values = self.forward_policy_value(public_features, private_features)
        # Normalize
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        log_prob = self.log_prob(actions, logits)
        entropy = self.entropy(logits)
        return values, log_prob, entropy

    def evaluate_actions_with_aux(self, public_features, private_features, actions):
        logits, p_aux1, p_aux2, p_aux3, values, v_aux = self.forward(public_features, private_features)
        # Normalize
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        log_prob = self.log_prob(actions, logits)
        entropy = self.entropy(logits)
        return values, log_prob, entropy, p_aux1, p_aux2, p_aux3, v_aux


class PolicyNet(nn.Module):
    def __init__(self, pv_net):
        super().__init__()
        self.pv_net = pv_net

    def forward(self, x):
        return self.pv_net.forward_policy(x)


class PolicyValueNet(nn.Module):
    def __init__(self, pv_net: PolicyValueNetWithAux):
        super().__init__()
        self.pv_net = pv_net

    def forward(self, x1, x2):
        return self.pv_net.forward_policy_value(x1, x2)
