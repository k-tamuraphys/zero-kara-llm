"""注意の性質を確かめる: 順序を知らない（置換同変性）、T^2 で増える計算量。"""

import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from capture import Capture

torch.manual_seed(0)
torch.set_num_threads(2)
cap = Capture()  # print の出力を out/ に保存する（capture.py）
OUT = Path(__file__).parent / "out"


def attention(X, W_q, W_k, W_v, causal: bool):
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    S = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    if causal:
        T = X.size(-2)
        S = S.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    return F.softmax(S, dim=-1) @ V


# ---- 1. 置換同変性: 入力の順を入れ替えると、出力も同じく入れ替わるだけ ----
# [snippet: permutation]
T, d = 5, 8
X = torch.randn(T, d)
W_q, W_k, W_v = (torch.randn(d, d) for _ in range(3))
perm = torch.tensor([3, 0, 4, 1, 2])  # 並べ替え方
Y = attention(X, W_q, W_k, W_v, causal=False)
Y_perm = attention(X[perm], W_q, W_k, W_v, causal=False)  # 入力を並べ替え
print("マスクなし: 並べ替えた入力の出力 = 出力の並べ替え?",
      torch.allclose(Y_perm, Y[perm], atol=1e-5))
Y = attention(X, W_q, W_k, W_v, causal=True)
Y_perm = attention(X[perm], W_q, W_k, W_v, causal=True)
print("因果マスクあり: 同じ関係が成り立つ?",
      torch.allclose(Y_perm, Y[perm], atol=1e-5))
past = torch.tensor([2, 0, 3, 1, 4])  # 最後の位置 4 は動かさず過去だけ入れ替え
Y_past = attention(X[past], W_q, W_k, W_v, causal=True)
print("因果マスクあり: 過去を入れ替えても最後の位置の出力は同じ?",
      torch.allclose(Y_past[-1], Y[-1], atol=1e-5))
# [end: permutation]
cap.save("permutation.txt")

# ---- 2. 系列長 T と計算時間・メモリ ----
B, H, d_head = 1, 4, 48  # 基盤モデルと同じ 4 ヘッド x 48 次元
rows = []
for T in [128, 256, 512, 1024, 2048, 4096]:
    q, k, v = (torch.randn(B, H, T, d_head) for _ in range(3))
    times = []
    for _ in range(7):  # CPU が混み合うとぶれるので 7 回測って中央値をとる
        t0 = time.perf_counter()
        S = q @ k.transpose(-2, -1) / math.sqrt(d_head)
        y = F.softmax(S, dim=-1) @ v
        times.append(time.perf_counter() - t0)
    ms = sorted(times)[len(times) // 2] * 1000
    mem = B * H * T * T * 4 / 2**20  # 注意の重み (B,H,T,T) の float32 の MiB
    rows.append((T, ms, mem))
    print(f"T={T:5d}  時間 {ms:8.2f} ms   重み行列 {mem:7.1f} MiB")
cap.save("cost.txt")
with open(OUT / "cost.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T", "ms", "mib"])
    for T, ms, mem in rows:
        w.writerow([T, f"{ms:.3f}", f"{mem:.2f}"])
