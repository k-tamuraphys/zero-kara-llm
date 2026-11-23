"""第 5 章: 初期値の大きさで、信号が層を進むときの大きさがどう変わるか。

幅 32・隠れ層 50 枚の ReLU の MLP に渦巻きデータを流し、
各層の出力の大きさ（成分の 2 乗平均の平方根）を測る。学習はしない。
    cd code && python ch05/init_stats.py > ch05/out/init_stats.txt
"""
import math

import torch
import torch.nn as nn

from ch05.deepmlp import load_spiral

DEPTH, WIDTH, SEEDS = 50, 32, 100
torch.set_num_threads(1)
X, Y, _, _ = load_spiral()


# [snippet: forward-rms]
@torch.no_grad()
def forward_sizes(std):
    """重みの標準偏差を std にして、各層の出力の 2 乗の平均を測る。"""
    sizes = []
    h = X                                        # (300, 2)
    n_in = 2
    for i in range(DEPTH):
        layer = nn.Linear(n_in, WIDTH)
        # 1 枚目だけは入力が 2 個なので、標準偏差を 1 にしておく
        nn.init.normal_(layer.weight, std=std if i > 0 else 1.0)
        nn.init.zeros_(layer.bias)
        h = torch.relu(layer(h))                 # (300, 32)
        sizes.append(h.pow(2).mean().item())     # 成分の 2 乗の平均
        n_in = WIDTH
    return torch.tensor(sizes)
# [end: forward-rms]


def mean_size(std):
    """乱数の種を変えて SEEDS 回測って平均し、平方根をとる。"""
    total = torch.zeros(DEPTH)
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        total += forward_sizes(std)
    return (total / SEEDS).sqrt().tolist()


STDS = (0.10, 0.20, 0.25, 0.30)
cols = {f"s{int(s * 100):03d}": mean_size(s) for s in STDS}
with open("ch05/out/init_stats.csv", "w") as f:
    f.write("layer," + ",".join(cols) + "\n")
    for i in range(DEPTH):
        f.write(f"{i + 1}," + ",".join(f"{v[i]:.4e}" for v in cols.values())
                + "\n")

print("各層の出力の大きさ（幅 32、ネットワーク 100 個の平均）")
print("標準偏差   第 1 層   第 10 層   第 50 層   1 層あたりの倍率   理論値")
for s, v in zip(STDS, cols.values()):
    ratio = (v[-1] / v[0]) ** (1 / (DEPTH - 1))
    print(f"  {s:.2f}  {v[0]:8.3f} {v[9]:10.2e} {v[49]:10.2e}"
          f" {ratio:12.3f} {s * math.sqrt(WIDTH / 2):12.2f}")
print()
print(f"He の初期化（幅 32）: sqrt(2 / 32) = {math.sqrt(2 / 32):.4f}")
print(f"PyTorch の標準（幅 32）: 1 / sqrt(3 * 32) = "
      f"{1 / math.sqrt(3 * 32):.4f}")
