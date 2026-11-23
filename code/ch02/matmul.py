"""行列積 @ と、「1 データ 1 行」の約束。"""

import torch

# [snippet: basic]
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])  # 形 (3, 2)
B = torch.tensor([[1.0, 0.0, 2.0, 0.0],
                  [0.0, 1.0, 0.0, 2.0]])  # 形 (2, 4)
print(A @ B)  # 行列積: (3, 2) @ (2, 4) → (3, 4)
v = torch.tensor([1.0, 1.0])  # 形 (2,)
print(A @ v)  # 行列とベクトル: (3, 2) @ (2,) → (3,)
print(A * A)  # 注意: * は要素ごとの積で、行列積ではない
# [end: basic]

print("---")
# [snippet: rows]
W = torch.tensor([[1.0, 0.0, -1.0],
                  [2.0, 1.0, 0.0]])  # 形 (2, 3)。3 個の数を 2 個の数に移す
b = torch.tensor([0.5, -0.5])  # 形 (2,)
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [1.0, 0.0, 0.0]])  # 形 (4, 3)。4 個のデータを行に並べた
for x in X:  # 数式どおり、1 個ずつ y = W x + b を計算する
    print(W @ x + b)
Y = X @ W.T + b  # 4 個まとめて: (4, 3) @ (3, 2) + (2,) → (4, 2)
print(Y)
# [end: rows]

print("---")
# [snippet: batch]
torch.manual_seed(0)
H = torch.randn(8, 5, 3)  # 8 個の文 × 5 個のトークン × 3 個の数
print((H @ W.T + b).shape)  # 40 個のベクトル全部に、同じ W と b を適用
C = torch.einsum("ik,kj->ij", A, B)  # C_ij = sum_k A_ik B_kj
print(torch.allclose(C, A @ B))  # @ と同じ結果か
# [end: batch]
