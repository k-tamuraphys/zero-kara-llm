"""3 件のデータで、損失・勾配・勾配降下法を手計算と照らし合わせる。"""

from pathlib import Path

import numpy as np

# [snippet: loss-grad-1d]
x = np.array([0.4, 0.8, 1.6])  # 文字数（百字単位）: 40 字, 80 字, 160 字
y = np.array([0.21, 0.42, 1.01])  # トークン数（百トークン単位）


def loss(w):
    """モデル f(x) = w x の平均二乗誤差。"""
    return np.mean((w * x - y) ** 2)


def grad(w):
    """損失を w で微分したもの。"""
    return np.mean(2 * (w * x - y) * x)
# [end: loss-grad-1d]


print("w      損失")
for w in [0.0, 0.3, 0.6, 0.9, 1.2]:
    print(f"{w:.1f}  {loss(w):.4f}")

print("損失 L(w) = a w^2 - c w + d の係数:")
print(f"a = {np.mean(x * x):.4f}, c = {2 * np.mean(x * y):.4f}, "
      f"d = {np.mean(y * y):.4f}")
w_star = np.sum(x * y) / np.sum(x * x)
print(f"最小点 w* = {w_star:.4f}, 最小値 = {loss(w_star):.4f}")

OUT = Path(__file__).parent / "out"
for eta in [0.4, 0.05, 0.8, 1.0]:
    print(f"--- 学習率 eta = {eta}")
    print("step  w        勾配      損失")
    w = 0.0
    with open(OUT / f"hand_gd_eta{eta}.csv", "w") as f:  # 図のための記録
        f.write("step,w,loss\n")
        for step in range(6):
            print(f"{step:4d}  {w:7.4f}  {grad(w):8.4f}  {loss(w):.4f}")
            f.write(f"{step},{w:.5f},{loss(w):.5f}\n")
            w = w - eta * grad(w)  # 勾配の逆向きに動かす
