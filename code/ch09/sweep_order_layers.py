"""確認: 位置の情報がない注意で、層の数とマスクの有無を変えて語順の影響を見る。

過去の 2 個のトークン（0 番目と 2 番目）を入れ替え、最後の位置の出力が
どれだけ変わるかを、1 層・2 層 × 因果マスクあり・なし の 4 通りで測る。
"""

import math

import torch

torch.manual_seed(0)
d, T = 16, 5


def attention(X, W, causal):
    """行列版の自己注意（位置の情報は入れない）。"""
    Q, K, V = X @ W[0], X @ W[1], X @ W[2]
    S = Q @ K.T / math.sqrt(d)
    if causal:
        S = S.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    return torch.softmax(S, dim=-1) @ V


X = torch.randn(T, d, dtype=torch.float64)
W1 = [torch.randn(d, d, dtype=torch.float64) / 4 for _ in range(3)]
W2 = [torch.randn(d, d, dtype=torch.float64) / 4 for _ in range(3)]
swap = [2, 1, 0, 3, 4]  # 0 番目と 2 番目を入れ替え、最後の位置は動かさない
for n_layer in [1, 2]:
    for causal in [True, False]:
        def f(X):
            Y = X + attention(X, W1, causal)  # 残差つき
            if n_layer == 2:
                Y = Y + attention(Y, W2, causal)
            return Y
        diff = (f(X)[-1] - f(X[swap])[-1]).abs().max().item()
        print(f"{n_layer} 層, 因果マスク{'あり' if causal else 'なし'}: "
              f"最後の位置の出力の差 {diff:.1e}")
