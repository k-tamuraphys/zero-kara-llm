"""第 18 章: 温度 tau を細かく変えて、KL 項の勾配の大きさを測る。

distill.py の後半と同じ作り方（64 位置 x 4,096 語彙、大きさ 3 の乱数の
ロジット）で、教師と生徒のロジットを 1 組だけ作り、tau だけを変える。
python ch18/sweep_tau_grad.py > ch18/out/sweep_tau_grad.txt
"""
import csv

import torch

from distill import distill_loss
from llmbook.corpus import DATA_DIR

OUT = DATA_DIR.parent / "ch18" / "out"
torch.manual_seed(0)
z_t = torch.randn(64, 4096) * 3                  # 教師のロジット（固定）
z_s0 = torch.randn(64, 4096) * 3                 # 生徒のロジット（固定）
y = torch.zeros(64, dtype=torch.long)            # alpha=1 なので使わない
rows = []
for tau in (1, 1.5, 2, 3, 4, 6, 8, 12, 16, 32):
    z_s = z_s0.clone().requires_grad_()
    _, kl, _ = distill_loss(z_s[None], z_t[None], y[None], tau, 1.0)
    g, = torch.autograd.grad(kl, z_s)
    rows.append(dict(tau=tau, grad=g.norm().item(),
                     grad_tau2=tau**2 * g.norm().item()))
    print(f"tau {tau:5.1f}  |grad KL| {rows[-1]['grad']:.6f}  "
          f"tau^2 * |grad KL| {rows[-1]['grad_tau2']:.6f}")
with open(OUT / "sweep_tau_grad.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tau", "grad", "grad_tau2"])
    w.writeheader()
    w.writerows(rows)
