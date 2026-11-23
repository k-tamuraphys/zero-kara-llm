"""なぜ √d_head で割るのか: 内積の分散と softmax の飽和を測る。"""

import csv
import math
from pathlib import Path

import torch

torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
N, T = 2000, 16  # 試行回数, キーの個数

rows = []
# [snippet: scaling]
for d in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    q = torch.randn(N, 1, d)  # 各成分が平均 0・分散 1 のクエリ
    k = torch.randn(N, T, d)  # 同じ分布のキー T 個
    s = (q * k).sum(dim=-1)  # 内積 (N, T)
    for name, z in [("raw", s), ("scaled", s / math.sqrt(d))]:
        a = torch.softmax(z, dim=-1)  # 注意の重み (N, T)
        # softmax の微分 (ヤコビ行列) diag(a) - a a^T の大きさ
        J = torch.diag_embed(a) - a.unsqueeze(-1) * a.unsqueeze(-2)
        rows.append({"d": d, "kind": name,
                     "var": z.var().item(),  # 内積の分散
                     "max_w": a.max(dim=-1).values.mean().item(),
                     "grad": J.norm(dim=(-2, -1)).mean().item()})
# [end: scaling]

with open(OUT / "scaling.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["d", "var_raw", "var_scaled", "maxw_raw", "maxw_scaled",
                "grad_raw", "grad_scaled"])
    for i in range(0, len(rows), 2):
        r, c = rows[i], rows[i + 1]
        w.writerow([r["d"], f"{r['var']:.3f}", f"{c['var']:.3f}",
                    f"{r['max_w']:.3f}", f"{c['max_w']:.3f}",
                    f"{r['grad']:.4f}", f"{c['grad']:.4f}"])
print(f"{'d':>4} {'分散(そのまま)':>10} {'分散(÷√d)':>9} "
      f"{'最大の重み(そのまま)':>12} {'最大の重み(÷√d)':>10}")
for i in range(0, len(rows), 2):
    r, c = rows[i], rows[i + 1]
    print(f"{r['d']:>4} {r['var']:>14.2f} {c['var']:>10.2f} "
          f"{r['max_w']:>20.3f} {c['max_w']:>15.3f}")
