"""計算ドリルの例だけから「足し算」を学ぶ（入力が 2 つの線形回帰）。"""

import random

import torch

from llmbook.arith import make_problem

# [snippet: data]
rng = random.Random(0)
rows = []
for _ in range(1000):
    question, answer = make_problem(rng, ops="+")  # 例: ("12+35=", "47")
    x1, x2 = question.rstrip("=").split("+")  # 足される 2 つの数
    rows.append((int(x1), int(x2), int(answer)))
print("データの先頭 3 件 (x1, x2, 答え):", rows[:3])
data = torch.tensor(rows, dtype=torch.float32) / 1000  # 千で割っておく
X, y = data[:, :2], data[:, 2]  # X: (N, 2)   y: (N,)
# [end: data]

# [snippet: train]
w = torch.zeros(2, requires_grad=True)  # 重みは入力ごとに 1 つ: (w1, w2)
b = torch.zeros(1, requires_grad=True)
eta = 0.5
for step in range(1001):
    y_hat = X @ w + b  # 全データの予測。形 (N,)
    loss = ((y_hat - y) ** 2).mean()
    if step in (0, 10, 100, 1000):
        print(f"step {step:4d}  w = ({w[0].item():.4f}, {w[1].item():.4f})"
              f"  b = {b.item():+.4f}  損失 = {loss.item():.2e}")
    loss.backward()
    with torch.no_grad():
        w -= eta * w.grad
        b -= eta * b.grad
    w.grad.zero_()
    b.grad.zero_()
# [end: train]

# [snippet: test]
x_new = torch.tensor([123.0, 456.0]) / 1000  # 学習データにない問題
print(f"123+456 の予測: {1000 * (x_new @ w + b).item():.1f}")
# [end: test]
