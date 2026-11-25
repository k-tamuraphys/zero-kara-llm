"""ReLU の折れ線で sin x を近似するとき、区間の数と誤差の関係を見る。

relu_approx.py と同じ作り方（区間 [0, 2π] を K 等分し、境目に ReLU を置く）で、
K を 2 から 256 まで 8 通りに変える。誤差は 2,001 点で測った最大値。
出力: ch04/out/sweep_relu_err.csv
"""

import math

import torch

torch.set_default_dtype(torch.float64)  # 小さな誤差まで正しく測るため


def relu_net(f, lo, hi, K):
    """relu_approx.py の relu_net と同じ。K 個の ReLU の折れ線を返す。"""
    t = torch.linspace(lo, hi, K + 1)  # 折れ目の位置
    slope = (f(t[1:]) - f(t[:-1])) / (t[1:] - t[:-1])  # 各区間の傾き
    c = slope.clone()
    c[1:] = slope[1:] - slope[:-1]  # 折れ目での傾きの変化

    def g(x):
        return f(t[0]) + torch.relu(x[:, None] - t[:-1]) @ c

    return g


x = torch.linspace(0.0, 2 * math.pi, 2001)
with open("ch04/out/sweep_relu_err.csv", "w") as out:
    out.write("K,err\n")
    prev = None
    for K in (2, 4, 8, 16, 32, 64, 128, 256):
        g = relu_net(torch.sin, 0.0, 2 * math.pi, K)
        err = (g(x) - torch.sin(x)).abs().max().item()
        out.write(f"{K},{err:.6e}\n")
        ratio = "" if prev is None else f"（前の {err / prev:.3f} 倍）"
        print(f"区間 {K:3d} 個: 誤差の最大値 {err:.6f}{ratio}")
        prev = err
