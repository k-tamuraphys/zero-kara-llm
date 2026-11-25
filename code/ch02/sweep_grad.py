"""数値微分と自動微分で、勾配を求める時間を比べる。

関数は f(x) = sum_i (x_i^2 + 3 x_i x_{i+1})（i = 1, ..., n-1）。
n = 2 なら、本文の f(x1, x2) = x1^2 + 3 x1 x2 と同じである。
変数の個数 n を 5 通りに変え、次の三つの時間を測る。
  f だけ: f を 1 回計算する
  自動微分: f を 1 回計算し、backward() を 1 回呼ぶ
  数値微分: 変数を 1 個ずつ前後に動かして、f を 2n 回計算する
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # speed.py と同じく 1 スレッドで測る

import csv  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(1)
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"


def best_time(f, repeat):
    """f() を repeat 回実行し、最も速かった回の秒数を返す。"""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


def f(x):
    """x は形 (n,)。値は数 1 個のテンソル。"""
    return (x[:-1] ** 2 + 3 * x[:-1] * x[1:]).sum()


def grad_auto(x0):
    """自動微分: f を 1 回計算し、記録を逆向きに 1 回たどる。"""
    x = x0.clone().requires_grad_(True)
    f(x).backward()
    return x.grad


def grad_numeric(x0, h=1e-6):
    """数値微分: 変数を 1 個ずつ前後に動かして、f を 2n 回計算する。"""
    x = x0.clone()
    g = torch.empty_like(x)
    for i in range(len(x)):
        x[i] += h
        fp = f(x)
        x[i] -= 2 * h
        fm = f(x)
        x[i] += h  # 元に戻す
        g[i] = (fp - fm) / (2 * h)
    return g


rows = []
print("       n   f だけ [ms]  自動微分 [ms]  (f の何倍)"
      "  数値微分 [ms]  差の最大")
for n in [10, 100, 1000, 10000, 100000]:
    # 数値微分の丸めの誤差を小さくするため、64 ビットの小数で計算する
    x0 = torch.randn(n, dtype=torch.float64)
    t_f = best_time(lambda: f(x0), 50) * 1e3
    t_a = best_time(lambda: grad_auto(x0), 50) * 1e3
    t_n = best_time(lambda: grad_numeric(x0), 3 if n <= 1000 else 1) * 1e3
    diff = (grad_auto(x0) - grad_numeric(x0)).abs().max().item()
    rows.append((n, t_f, t_a, t_n, diff))
    print(f"{n:8d} {t_f:12.4f} {t_a:14.4f} {t_a / t_f:9.1f} 倍"
          f" {t_n:14.1f} {diff:9.1e}")

with open(OUT / "sweep_grad.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["n", "f_ms", "auto_ms", "numeric_ms", "maxdiff"])
    for n, t_f, t_a, t_n, diff in rows:
        w.writerow([n, f"{t_f:.5f}", f"{t_a:.5f}", f"{t_n:.3f}",
                    f"{diff:.2e}"])
