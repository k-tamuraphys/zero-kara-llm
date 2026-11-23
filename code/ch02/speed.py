"""for 文と配列演算の速度を、行列積で比べる。"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # 公平のため、NumPy も 1 スレッドにする
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(1)  # PyTorch も 1 スレッドにする
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"


# [snippet: loop]
def matmul_loop(A, B):
    """行列積 C_ij = sum_k A_ik B_kj を、3 重の for 文で計算する。"""
    n, m, p = len(A), len(B), len(B[0])  # A は n 行 m 列、B は m 行 p 列
    C = [[0.0] * p for _ in range(n)]  # 答えを入れる n 行 p 列の表
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C
# [end: loop]


# [snippet: timer]
def best_time(f, repeat):
    """f() を repeat 回実行し、最も速かった回の秒数を返す。"""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)
# [end: timer]


# [snippet: compare]
n = 64
A_t, B_t = torch.randn(n, n), torch.randn(n, n)  # 乱数で埋めたテンソル
A_np, B_np = A_t.numpy(), B_t.numpy()  # 同じ数を NumPy の配列として
A_list, B_list = A_t.tolist(), B_t.tolist()  # 同じ数を Python のリストとして

C_loop = matmul_loop(A_list, B_list)  # for 文
C_np = A_np @ B_np  # NumPy
C_t = A_t @ B_t  # PyTorch
diff = (torch.tensor(C_loop) - C_t).abs().max().item()
print(f"for 文と PyTorch の答えの差（最大） {diff:.1e}")
# [end: compare]

# 大きさ n を変えて測り、グラフ用に CSV へ保存する。
print("     n   for 文 [ms]   NumPy [ms]  PyTorch [ms]   for 文 / PyTorch")
rows = []
for n in [8, 16, 32, 64, 128, 256]:
    A_t, B_t = torch.randn(n, n), torch.randn(n, n)
    A_np, B_np = A_t.numpy(), B_t.numpy()
    A_list, B_list = A_t.tolist(), B_t.tolist()
    t_loop = best_time(lambda: matmul_loop(A_list, B_list),
                       repeat=5 if n <= 128 else 2)
    t_np = best_time(lambda: A_np @ B_np, repeat=30)
    t_pt = best_time(lambda: A_t @ B_t, repeat=30)
    rows.append((n, t_loop * 1e3, t_np * 1e3, t_pt * 1e3))
    print(f"{n:6d} {t_loop * 1e3:12.3f} {t_np * 1e3:12.4f} {t_pt * 1e3:13.4f}"
          f" {t_loop / t_pt:14,.0f} 倍")
with open(OUT / "speed_matmul.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "loop_ms", "numpy_ms", "torch_ms"])
    for n, tl, tn, tp in rows:
        w.writerow([n, f"{tl:.5f}", f"{tn:.5f}", f"{tp:.5f}"])

print("---")
# ミニGPT が 1 回の予測の最後に行う行列積と同じ大きさ:
# (128, 192) と (192, 4096)。掛け算の回数は 128*192*4096 ≒ 1 億回。
# [snippet: real-size]
H_t, E_t = torch.randn(128, 192), torch.randn(192, 4096)
H_list, E_list = H_t.tolist(), E_t.tolist()
t_loop = best_time(lambda: matmul_loop(H_list, E_list), repeat=2)
t_pt = best_time(lambda: H_t @ E_t, repeat=30)
print(f"for 文   {t_loop:8.2f} 秒")
print(f"PyTorch  {t_pt * 1e3:8.2f} ミリ秒")
print(f"for 文は PyTorch の {t_loop / t_pt:,.0f} 倍の時間がかかる")
# [end: real-size]
