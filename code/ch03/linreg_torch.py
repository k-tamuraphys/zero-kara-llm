"""線形回帰を PyTorch の自動微分で解く（勾配の式を自分で導かない版）。"""

from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).parent / "out"
data = np.loadtxt(OUT / "char_token_train.csv", delimiter=",", skiprows=1)

# [snippet: setup]
x = torch.tensor(data[:, 0] / 100, dtype=torch.float32)  # 形 (N,)
y = torch.tensor(data[:, 1] / 100, dtype=torch.float32)  # 形 (N,)
w = torch.tensor(0.0, requires_grad=True)  # 勾配を求めたいパラメータ
b = torch.tensor(0.0, requires_grad=True)

loss = ((w * x + b - y) ** 2).mean()  # 損失を「計算するだけ」
loss.backward()  # 自動微分: w.grad と b.grad に勾配が入る
print(f"自動微分   dL/dw = {w.grad.item():.6f}  dL/db = {b.grad.item():.6f}")
err = (w * x + b - y).detach()  # 手で導いた式でも計算して比べる
print(f"手で導いた式 dL/dw = {2 * (err * x).mean().item():.6f}  "
      f"dL/db = {2 * err.mean().item():.6f}")
# [end: setup]
w.grad.zero_()
b.grad.zero_()

# [snippet: loop]
eta = 0.1
for step in range(1001):
    loss = ((w * x + b - y) ** 2).mean()  # (1) 予測して損失を計算
    if step in (0, 10, 100, 1000):
        print(f"step {step:4d}  w = {w.item():.4f}  b = {b.item():+.4f}  "
              f"損失 = {loss.item():.5f}")
    loss.backward()  # (2) 勾配を計算
    with torch.no_grad():  # (3) パラメータを更新（この計算は微分の対象外）
        w -= eta * w.grad
        b -= eta * b.grad
    w.grad.zero_()  # (4) 次のステップのために勾配を 0 に戻す
    b.grad.zero_()
# [end: loop]
