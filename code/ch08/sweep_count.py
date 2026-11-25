"""第 8 章: 「数える方法」を工夫すると、どこまで下がるか（MLP との比較）。

直前 k 個（k = 1, 2, 3）を数える表に、標準的な工夫を一つ入れる。
数えた回数から一定の数 D を引き、余った確率を「直前 k-1 個の表」に回す
（補間つきの絶対割引。短い文脈の表で、見たことのない並びを補う）。
実行: cd code && python ch08/sweep_count.py > ch08/out/sweep_count.txt
      （1 分ほど）
出力: ch08/out/sweep_count.txt, ch08/out/sweep_count.csv,
      ch08/out/sweep_count_best.csv（k ごとに、全体の損失が最小の D の行）
"""
import math
from collections import Counter, defaultdict

import torch

from capture import OUT
from mlp_lm import V, split_mean, train, val

K_MAX = 3
seq, vseq = train.tolist(), val.tolist()

# 並び（文脈 h と次の w）の回数、文脈の回数、文脈のあとに来た種類の数
cnt = [Counter() for _ in range(K_MAX + 1)]
for k in range(K_MAX + 1):
    for i in range(k, len(seq)):
        cnt[k][tuple(seq[i - k:i + 1])] += 1
ctx = [Counter() for _ in range(K_MAX + 1)]
kinds = [defaultdict(int) for _ in range(K_MAX + 1)]
for k in range(K_MAX + 1):
    for g, c in cnt[k].items():
        ctx[k][g[:-1]] += c
        kinds[k][g[:-1]] += 1


def prob(h, w, D):
    """直前の並び h（長さ k）のあとに w が来る確率（短い文脈から順に）。"""
    p = 1.0 / V                                 # いちばん下は一様分布
    for k in range(len(h) + 1):
        hk = tuple(h[len(h) - k:])              # 直前 k 個
        n = ctx[k].get(hk, 0)
        if n == 0:                              # この文脈は一度も見ていない
            continue
        c = cnt[k].get(hk + (w,), 0)
        p = max(c - D, 0) / n + D * kinds[k][hk] / n * p
    return p


rows = []
for k in range(1, K_MAX + 1):
    for D in [0.5, 0.75, 0.9]:
        losses = torch.tensor([-math.log(prob(vseq[i - k:i], vseq[i], D))
                               for i in range(k, len(vseq))])
        a, b, c = split_mean(losses, k)
        rows.append((k, D, a, b, c))
        print(f"直前 {k} 個を数える  D={D:.2f}  全体 {a:.3f}  小説 {b:.3f}"
              f"  ドリル {c:.3f}")
with open(OUT / "sweep_count.csv", "w") as f:
    f.write("n,D,val,val_novel,val_drill\n")
    for k, D, a, b, c in rows:
        f.write(f"{k},{D},{a:.4f},{b:.4f},{c:.4f}\n")
best = [min((r for r in rows if r[0] == k), key=lambda r: r[2])
        for k in range(1, K_MAX + 1)]       # D は検証データで選ぶ（α と同じ）
with open(OUT / "sweep_count_best.csv", "w") as f:
    f.write("n,D,val,val_novel,val_drill\n")
    for k, D, a, b, c in best:
        f.write(f"{k},{D},{a:.4f},{b:.4f},{c:.4f}\n")
