"""第 5 章（追加の確認）: PyTorch の既定の初期値で、バイアスの扱いを変える。

init_stats.py と同じ幅 32・隠れ層 50 枚の ReLU の MLP に渦巻きデータを流す。
重みは nn.Linear の既定の初期値のまま、バイアスを (a) 0 にする、
(b) 既定の乱数のままにする、の 2 通りで、各層の出力を測る。学習はしない。
    cd code && python ch05/sweep_bias.py > ch05/out/sweep_bias.txt
"""
import torch
import torch.nn as nn

from ch05.deepmlp import load_spiral

torch.set_num_threads(1)
X, _, _, _ = load_spiral()
DEPTH, WIDTH, SEEDS = 50, 32, 20


@torch.no_grad()
def measure(zero_bias):
    """各層の (出力の大きさ, データの間での散らばり) を、種 SEEDS 個で平均。"""
    size = torch.zeros(DEPTH)
    spread = torch.zeros(DEPTH)
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        h, n_in = X, 2
        for i in range(DEPTH):
            layer = nn.Linear(n_in, WIDTH)           # 既定の初期値
            if zero_bias:
                nn.init.zeros_(layer.bias)
            h = torch.relu(layer(h))                 # (300, 32)
            size[i] += h.pow(2).mean().sqrt()        # 2 乗の平均の平方根
            spread[i] += h.std(dim=0).mean()         # 入力による違い
            n_in = WIDTH
    return size / SEEDS, spread / SEEDS


print("既定の初期値、幅 32（種 20 個の平均）")
print("バイアス      層   出力の大きさ   データの間での散らばり")
for zero_bias, name in ((True, "0 にする"), (False, "既定の乱数")):
    size, spread = measure(zero_bias)
    for i in (0, 9, 49):
        print(f"{name:<10s} {i + 1:4d}   {size[i]:10.3g}   {spread[i]:14.3g}")
