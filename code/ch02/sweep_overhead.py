"""ごく小さな行列積 1 回の時間を、NumPy と PyTorch で測る。

n 行 n 列どうしの積で、n を 1 から 64 まで 7 通りに変える。
n = 1 の掛け算は 1 回だけなので、その時間は、ほぼ
「関数を呼び出して計算を始めるまでの準備の時間」である。
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # speed.py と同じく 1 スレッドで測る
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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


def per_call(f, inner=1000, repeat=20):
    """f() を inner 回続けて呼んだ時間を測り、1 回あたりの秒数を返す。"""
    def many():
        for _ in range(inner):
            f()
    return best_time(many, repeat) / inner


rows = []
print("    n  掛け算の回数  NumPy [µs]  PyTorch [µs]")
for n in [1, 2, 4, 8, 16, 32, 64]:
    A_t, B_t = torch.randn(n, n), torch.randn(n, n)  # float32
    A_np, B_np = A_t.numpy(), B_t.numpy()  # 同じ数を NumPy で
    t_np = per_call(lambda: A_np @ B_np) * 1e6
    t_pt = per_call(lambda: A_t @ B_t) * 1e6
    rows.append((n, t_np, t_pt))
    print(f"{n:5d} {n ** 3:12d} {t_np:11.2f} {t_pt:13.2f}")

with open(OUT / "sweep_overhead.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "numpy_us", "torch_us"])
    for n, a, c in rows:
        w.writerow([n, f"{a:.4f}", f"{c:.4f}"])
