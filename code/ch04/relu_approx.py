"""ReLU を K 個足した折れ線で sin 曲線を近似する（万能近似定理の直感）。"""

import math

import torch


# [snippet: approx]
def relu_net(f, lo, hi, K):
    """区間 [lo, hi] を K 等分し、f を折れ線でなぞる ReLU ネットを作る。"""
    t = torch.linspace(lo, hi, K + 1)  # 折れ目の位置（K+1 個）
    slope = (f(t[1:]) - f(t[:-1])) / (t[1:] - t[:-1])  # 各区間の傾き (K,)
    c = slope.clone()
    c[1:] = slope[1:] - slope[:-1]  # 折れ目での「傾きの変化」

    def g(x):  # x: (N,)
        h = torch.relu(x[:, None] - t[:-1])  # 隠れ層: K 個の ReLU (N, K)
        return f(t[0]) + h @ c  # 出力層: 重み c で足し合わせる

    return g
# [end: approx]


x = torch.linspace(0.0, 2 * math.pi, 201)
cols = {"x": x, "f": torch.sin(x)}
for K in (4, 8, 16, 32, 64):
    g = relu_net(torch.sin, 0.0, 2 * math.pi, K)
    err = (g(x) - torch.sin(x)).abs().max().item()
    print(f"ReLU {K:2d} 個: 誤差の最大値 {err:.4f}")
    cols[f"g{K}"] = g(x)
with open("ch04/out/relu_approx.csv", "w") as out:
    out.write(",".join(cols) + "\n")
    for i in range(len(x)):
        out.write(",".join(f"{v[i].item():.5f}" for v in cols.values()) + "\n")
