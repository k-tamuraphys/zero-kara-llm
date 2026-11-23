"""GELU と ReLU の曲線を CSV にし、本文の数値例を確かめる。

実行: python ch10/gelu_curve.py
出力: ch10/out/gelu.csv, gelu.txt
"""

from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).parent / "out"

u = torch.linspace(-4, 4, 161)  # 0.05 刻み
with open(OUT / "gelu.csv", "w", encoding="utf-8") as f:
    f.write("u,gelu,relu\n")
    for a, g, r in zip(u.tolist(), F.gelu(u).tolist(), F.relu(u).tolist()):
        f.write(f"{a:.2f},{g:.4f},{r:.4f}\n")

normal = torch.distributions.Normal(0.0, 1.0)
lines = []
for a in [3.0, 0.5, -0.5, -3.0]:
    t = torch.tensor(a)
    lines.append(f"u = {a:4.1f}: Phi(u) = {normal.cdf(t):.4f}, "
                 f"GELU(u) = {F.gelu(t):.4f}, ReLU(u) = {F.relu(t):.1f}")
(OUT / "gelu.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
