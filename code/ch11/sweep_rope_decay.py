"""RoPE の内積が位置の差でどう変わるかを、q と k の選び方を変えて比べる。

図 11.4(b) の「全成分 1」のほかに、ランダムな q と同じ向きの k（k = q）と、
互いに無関係なランダムな q, k を、それぞれ 200 組ずつ平均する（図用の CSV）。
使い方: python ch11/sweep_rope_decay.py > ch11/out/sweep_rope_decay.txt
"""
import sys

import torch

sys.path.insert(0, "ch11")
from modern_gpt import apply_rope, rope_tables

torch.manual_seed(0)
dh, n_pair = 64, 200
cos, sin = rope_tables(dh, 512)
ones = torch.ones(dh)
q = torch.randn(n_pair, dh)                  # ランダムな q（200 組）
k = torch.randn(n_pair, dh)                  # q と無関係なランダムな k


def score(a, b, dist):
    """位置 dist の a と位置 0 の b を回して内積をとり、長さの積で割る。"""
    ra = apply_rope(a, cos[dist], sin[dist])
    rb = apply_rope(b, cos[0], sin[0])
    return (ra * rb).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


rows = []
for dist in range(0, 257, 2):
    rows.append((dist,
                 score(ones, ones, dist).item(),        # 全成分 1（図 11.4）
                 score(q, q, dist).mean().item(),       # 同じ向き（k = q）
                 score(q, k, dist).mean().item(),       # 無関係（200 組平均）
                 score(q[0], k[0], dist).item()))       # 無関係（1 組だけ）
with open("ch11/out/sweep_rope_decay.csv", "w") as f:
    f.write("dist,ones,same,indep_mean,indep_one\n")
    for r in rows:
        f.write(",".join(f"{x:.5f}" if i else str(x)
                         for i, x in enumerate(r)) + "\n")
for r in rows:
    if r[0] in (0, 2, 16, 64, 128, 256):
        print(f"差 {r[0]:3d}: 全成分1 {r[1]:+.3f}  同じ向き {r[2]:+.3f}  "
              f"無関係の平均 {r[3]:+.3f}  無関係の1組 {r[4]:+.3f}")
