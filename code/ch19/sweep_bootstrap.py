"""ブートストラップの分布を図にするための度数分布（stats.py の続き）。

stats.py と同じ 1,000 問（種 777）の正誤から、重複を許して選び直すことを
1 万回くり返し、正答率の度数分布と、式から予想される正規分布を残す。
"""

import csv
import math

import numpy as np

from common import evaluate

correct = np.array([r["correct"] for r in evaluate("sft", 1000, 777)],
                   dtype=float)
n = len(correct)
rng = np.random.default_rng(0)  # stats.py の bootstrap_ci と同じ種
picks = rng.integers(0, n, size=(10_000, n))
accs = correct[picks].mean(axis=1) * 100          # 選び直すたびの正答率（%）
lo, hi = np.percentile(accs, 2.5), np.percentile(accs, 97.5)
p = correct.mean()
se = math.sqrt(p * (1 - p) / n) * 100             # 式による標準誤差（%）
print(f"選び直した正答率: 平均 {accs.mean():.2f}%  標準偏差 {accs.std():.2f}%"
      f"（式: {se:.2f}%）")
print(f"下から 2.5% の位置 {lo:.1f}%  97.5% の位置 {hi:.1f}%")

# 幅 0.5 ポイントの度数分布（中心 c、範囲 [c - 0.25, c + 0.25)）
with open("ch19/out/sweep_bootstrap.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["acc", "count", "normal"])
    for c in np.arange(78.0, 87.01, 0.5):
        count = int(((accs >= c - 0.25) & (accs < c + 0.25)).sum())
        z = (c - 100 * p) / se
        normal = 10_000 * 0.5 / se * math.exp(-z * z / 2) / math.sqrt(
            2 * math.pi)
        w.writerow([f"{c:.1f}", count, f"{normal:.1f}"])
with open("ch19/out/sweep_bootstrap_ci.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["lo", "hi", "p"])
    w.writerow([f"{lo:.2f}", f"{hi:.2f}", f"{100 * p:.2f}"])
