"""線形モデル（第 3 章の softmax 回帰）を渦巻きデータに当てはめる。

結果: 正解率は 5 割ほどで頭打ちになる。
図用に、データ点と決定領域を CSV に保存する。
"""

import torch
import torch.nn.functional as F

from data import make_spiral

torch.manual_seed(0)
X, Y = make_spiral(n_per_class=100, seed=0)  # 訓練用 300 点
X, Y = torch.tensor(X), torch.tensor(Y)

W = torch.zeros(3, 2, requires_grad=True)  # 重み（3 クラス × 2 次元）
b = torch.zeros(3, requires_grad=True)  # バイアス
for step in range(1001):
    z = X @ W.T + b  # (300, 3) ロジット
    loss = F.cross_entropy(z, Y)
    W.grad, b.grad = None, None
    loss.backward()
    with torch.no_grad():
        W -= 1.0 * W.grad
        b -= 1.0 * b.grad
    if step % 250 == 0:
        acc = (z.argmax(dim=1) == Y).float().mean().item()
        print(f"step {step:4d}  損失 {loss.item():.4f}  正解率 {acc:.3f}")

# 図用: データ点
with open("ch04/out/spiral_train.csv", "w") as f:
    f.write("x1,x2,label\n")
    for (x1, x2), y in zip(X.tolist(), Y.tolist()):
        f.write(f"{x1:.4f},{x2:.4f},{y}\n")
# 図用: 格子点ごとの予測クラス（決定領域）
n = 60
g = torch.linspace(-1.15, 1.15, n)
with torch.no_grad(), open("ch04/out/linear_regions.csv", "w") as f:
    f.write("x1,x2,c\n")
    for x2 in g.tolist():
        for x1 in g.tolist():
            c = (torch.tensor([x1, x2]) @ W.T + b).argmax().item()
            f.write(f"{x1:.4f},{x2:.4f},{c}\n")
