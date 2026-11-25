"""第 5 章（追加のスイープ）: 初期値の標準偏差を変えて学習させる。

隠れ層 10 枚・幅 32 の MLP（正規化・残差接続なし）で、間の層の重みの
標準偏差 s を 0.10〜0.60 の 8 通りに変え、SGD（学習率 0.1）で
1,000 ステップ学習させる。乱数の種 0〜2 の 3 回。
（学習率を変える実験は sweep_lr_depth.py）
    cd code && OMP_NUM_THREADS=2 python ch05/sweep_init.py \
        > ch05/out/sweep_init.txt
csv は ch05/out/sweep_init_curves.csv と sweep_init_final.csv に書く。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch05.deepmlp import DeepMLP, load_spiral

torch.set_num_threads(1)
X, Y, _, _ = load_spiral()
STEPS = 1000
SEEDS = (0, 1, 2)


def set_std_(model, std):
    """最初の層は標準偏差 1、間の層は std にする（バイアスは 0）。"""
    nn.init.normal_(model.first.weight, std=1.0)   # 入力 2 個: sqrt(2/2)
    nn.init.zeros_(model.first.bias)
    for layer in model.layers:
        nn.init.normal_(layer.weight, std=std)
        nn.init.zeros_(layer.bias)


def train(model, lr):
    """全データ（300 点）の勾配で STEPS 回更新し、損失の列を返す。"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    curve = []
    for step in range(STEPS):
        loss = F.cross_entropy(model(X), Y)
        curve.append(loss.item())
        if not math.isfinite(curve[-1]):     # 発散したら打ち切る
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return curve


@torch.no_grad()
def accuracy(model):
    return (model(X).argmax(dim=1) == Y).float().mean().item()


# 標準偏差を変える（隠れ層 10 枚、学習率 0.1）
STDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60)
print("隠れ層 10 枚、SGD（学習率 0.1）、1,000 ステップ")
print("標準偏差  学習できた回数  最後の訓練損失（種 0, 1, 2）")
curves, finals = {}, []
for s in STDS:
    losses, ok = [], 0
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = DeepMLP(10, init="default")
        set_std_(model, s)
        curve = train(model, 0.1)
        if seed == 0:
            curves[s] = curve
        last = curve[-1] if len(curve) == STEPS else float("nan")
        ok += int(math.isfinite(last) and accuracy(model) >= 0.99)
        losses.append(last)
    finals.append((s, ok, losses))
    print(f"  {s:.2f}   {ok} 回   " + "  ".join(f"{v:.4f}" for v in losses))

with open("ch05/out/sweep_init_curves.csv", "w") as f:
    f.write("step," + ",".join(f"s{int(s * 100):03d}" for s in STDS) + "\n")
    for i in range(0, STEPS, 5):
        f.write(f"{i}," + ",".join(
            f"{curves[s][i]:.5f}" if i < len(curves[s]) else "nan"
            for s in STDS) + "\n")
with open("ch05/out/sweep_init_final.csv", "w") as f:
    f.write("std,ok,mean_loss\n")
    for s, ok, losses in finals:
        good = [v for v in losses if math.isfinite(v)]
        mean = sum(good) / len(good) if good else float("nan")
        f.write(f"{s},{ok},{mean:.5f}\n")
