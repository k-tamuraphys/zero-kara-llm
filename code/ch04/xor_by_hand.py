"""ReLU のニューロン 2 個を手で設計して、XOR を解く。"""

import torch

# [snippet: xor]
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  # 4 つの入力
W1 = torch.tensor([[1., 1.], [1., 1.]])  # 隠れ層の重み (2, 2)
b1 = torch.tensor([0., -1.])  # 隠れ層のバイアス
w2 = torch.tensor([1., -2.])  # 出力の重み
U = X @ W1.T + b1  # 線形変換 (4, 2)
H = torch.relu(U)  # 活性化関数を通す (4, 2)
z = H @ w2  # 出力 (4,)
for x, u, h, zi in zip(X.tolist(), U.tolist(), H.tolist(), z.tolist()):
    print(f"x={x}  u={u}  h={h}  z={zi}")
# [end: xor]
