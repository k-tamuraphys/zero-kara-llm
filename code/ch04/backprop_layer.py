"""2 層の MLP の逆伝播を、行列の式で手書きし、autograd と比べる。"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(4)  # 入力（4 次元）
y = torch.tensor(1)  # 正解クラス
W1 = torch.randn(3, 4, requires_grad=True)
b1 = torch.randn(3, requires_grad=True)
W2 = torch.randn(2, 3, requires_grad=True)
b2 = torch.randn(2, requires_grad=True)

# [snippet: manual]
# --- 順伝播（途中の値をすべて取っておく）
u1 = W1 @ x + b1  # (3,)
h1 = torch.relu(u1)  # (3,)
z = W2 @ h1 + b2  # (2,) ロジット
loss = F.cross_entropy(z, y)

# --- 逆伝播（出力側から 1 段ずつ。d◯ は「損失を ◯ で微分したもの」）
with torch.no_grad():
    dz = torch.softmax(z, dim=0) - F.one_hot(y, 2)  # p - y（第 3 章）
    dW2 = torch.outer(dz, h1)  # (2, 3) = 上流の勾配 × 入力
    db2 = dz  # (2,)
    dh1 = W2.T @ dz  # (3,) 下の層へ渡す勾配
    du1 = dh1 * (u1 > 0)  # ReLU: 正だった成分だけ通す
    dW1 = torch.outer(du1, x)  # (3, 4)
    db1 = du1  # (3,)
# [end: manual]

loss.backward()  # autograd に同じ勾配を計算させる
for name, t in [("x  ", x), ("u1 ", u1), ("h1 ", h1), ("dz ", dz),
                ("dh1", dh1), ("du1", du1)]:
    print(f"{name} =", [round(v, 3) + 0.0 for v in t.tolist()])
print("dW1 =")
for row in dW1.tolist():
    print("   ", [round(v, 3) + 0.0 for v in row])
for name, mine, p in [("W1", dW1, W1), ("b1", db1, b1),
                      ("W2", dW2, W2), ("b2", db2, b2)]:
    diff = (mine - p.grad).abs().max().item()
    shape = tuple(mine.shape)
    print(f"{name}: 形 {shape}  autograd との差の最大値 {diff:.1e}")
