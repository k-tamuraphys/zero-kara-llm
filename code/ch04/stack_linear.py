"""線形の層を何枚重ねても、1 枚の線形の層と同じであることを確かめる。"""

import torch

torch.manual_seed(0)
# [snippet: stack]
x = torch.randn(5, 2)  # 入力 5 個（1 行が 1 データ）
W, b = torch.eye(2), torch.zeros(2)  # まとめた 1 枚ぶん（最初は何もしない層）
h = x
for layer in range(10):  # 線形の層を 10 枚重ねる
    Wl, bl = torch.randn(2, 2) * 0.7, torch.randn(2)
    h = h @ Wl.T + bl  # この層の出力
    W, b = Wl @ W, Wl @ b + bl  # ここまでの層を 1 枚にまとめる
one = x @ W.T + b  # まとめた 1 枚だけを通した結果
print("10 枚重ねた出力:", h[0])
print("1 枚にまとめた出力:", one[0])
print("差の最大値:", (h - one).abs().max().item())
# [end: stack]
