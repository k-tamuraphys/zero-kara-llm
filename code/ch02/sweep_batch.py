"""1 個ずつの W @ x + b と、まとめての X @ W.T + b の速さを比べる。

データの個数 N を 6 通りに変え、データ 1 個あたりの時間を測る。
1 個ずつの計算は、2 通り測る。数式どおりの W @ x + b と、
x を 1 行の行列にして X @ W.T + b と同じ式で計算するものである。
重み行列 W は、基盤モデルにある形 (576, 192) と同じ大きさにした。
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


def one_by_one(W, b, X):
    """数式どおり、データを 1 個ずつ取り出して W x + b を計算する。"""
    return torch.stack([W @ x + b for x in X])  # (N, 576)


def one_by_one_row(W, b, X):
    """1 個ずつだが、x を 1 行の行列 (1, 192) にして、同じ式で計算する。"""
    return torch.cat([x[None] @ W.T + b for x in X])  # (N, 576)


def batched(W, b, X):
    """1 データ 1 行の約束で、N 個まとめて計算する。"""
    return X @ W.T + b  # (N, 192) @ (192, 576) + (576,) → (N, 576)


W = torch.randn(576, 192)  # 192 個の数を 576 個の数に移す重み行列
b = torch.randn(576)  # バイアス
rows = []
print("     N  1 個ずつ W@x  1 個ずつ 1 行  まとめて  [µs/個]"
      "  倍率  一致")
for N in [1, 4, 16, 64, 256, 1024]:
    X = torch.randn(N, 192)  # N 個のデータを行に並べる
    Y = batched(W, b, X)
    same = (torch.allclose(one_by_one(W, b, X), Y, atol=1e-4)
            and torch.allclose(one_by_one_row(W, b, X), Y, atol=1e-4))
    rep = 200 if N <= 16 else 30
    t1 = best_time(lambda: one_by_one(W, b, X), rep) / N * 1e6
    t1r = best_time(lambda: one_by_one_row(W, b, X), rep) / N * 1e6
    t2 = best_time(lambda: batched(W, b, X), rep) / N * 1e6
    rows.append((N, t1, t1r, t2))
    print(f"{N:6d} {t1:12.2f} {t1r:14.2f} {t2:9.3f} {' ' * 7}"
          f"{t1 / t2:6.1f} 倍  {same}")

with open(OUT / "sweep_batch.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "loop_us", "looprow_us", "batch_us", "ratio"])
    for N, t1, t1r, t2 in rows:
        w.writerow([N, f"{t1:.4f}", f"{t1r:.4f}", f"{t2:.4f}",
                    f"{t1 / t2:.2f}"])
