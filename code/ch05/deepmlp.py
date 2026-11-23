"""第 5 章: 深い MLP。初期化・正規化・残差接続を切り替えられる。

ほかのスクリプトが読み込んで使う（これ自体は実行しない）。
"""
import math

import torch
import torch.nn as nn

from ch04.data import make_spiral


def load_spiral():
    """第 4 章と同じ渦巻きデータ（訓練 300 点、テスト 300 点）。"""
    X, Y = make_spiral(n_per_class=100, seed=0)
    Xt, Yt = make_spiral(n_per_class=100, seed=1)
    return (torch.tensor(X), torch.tensor(Y),
            torch.tensor(Xt), torch.tensor(Yt))


# [snippet: he-init]
def he_init_(layer):
    """He の初期化。重みを標準偏差 sqrt(2 / 入力の数) の正規分布にする。"""
    std = math.sqrt(2.0 / layer.in_features)
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    nn.init.zeros_(layer.bias)       # バイアスは 0 から始める
# [end: he-init]


class DeepMLP(nn.Module):
    """隠れ層を depth 枚重ねた MLP（活性化関数は ReLU）。"""

    # [snippet: deepmlp-init]
    def __init__(self, depth, width=32, init="he", norm=False,
                 residual=False):
        super().__init__()
        self.first = nn.Linear(2, width)             # 入力 2 → 幅
        self.layers = nn.ModuleList(                 # 幅 → 幅 を depth-1 枚
            [nn.Linear(width, width) for _ in range(depth - 1)])
        self.norms = nn.ModuleList(                  # 各層に付ける正規化
            [nn.LayerNorm(width) if norm else nn.Identity()
             for _ in range(depth - 1)])
        self.last_norm = (nn.LayerNorm(width) if norm and residual
                          else nn.Identity())
        self.out = nn.Linear(width, 3)               # 幅 → 3 クラス
        self.residual = residual
        if init == "he":
            for layer in [self.first, *self.layers]:
                he_init_(layer)
    # [end: deepmlp-init]

    # [snippet: deepmlp-forward]
    def forward(self, x):                            # x: (N, 2)
        h = torch.relu(self.first(x))                # (N, 幅)
        for layer, norm in zip(self.layers, self.norms):
            if self.residual:
                h = h + torch.relu(layer(norm(h)))   # 残差接続: 足し込む
            else:
                h = torch.relu(norm(layer(h)))       # ふつうの層: 置き換える
        return self.out(self.last_norm(h))           # (N, 3)
    # [end: deepmlp-forward]


@torch.no_grad()
def evaluate(model, X, Y):
    """損失と正解率を返す。"""
    import torch.nn.functional as F
    logits = model(X)
    loss = F.cross_entropy(logits, Y).item()
    acc = (logits.argmax(dim=1) == Y).float().mean().item()
    return loss, acc
