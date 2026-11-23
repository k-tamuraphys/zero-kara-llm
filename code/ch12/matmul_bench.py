"""行列積の速さを測る: for 文 vs. BLAS、行列サイズ、スレッド数、float32 と float64。"""
import time
import torch

torch.manual_seed(0)


# [snippet: naive]
def matmul_loop(A, B):
    """教科書どおりの 3 重ループ。C_ij = sum_k A_ik B_kj。"""
    n, m, p = len(A), len(B), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C
# [end: naive]


# [snippet: bench]
def bench(fn, repeat=5):
    """fn() の所要時間（秒）の中央値。1 回目は準備運動として捨てる。"""
    fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        # GPU なら、ここで torch.cuda.synchronize() が必要（本文参照）
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def gflops(n, sec):
    return 2 * n**3 / sec / 1e9    # n×n 行列積は 2n^3 FLOPs
# [end: bench]


torch.set_num_threads(1)
print("== for 文 vs. torch.matmul（1 スレッド, float64）==")
for n in (64, 128):
    A = torch.randn(n, n, dtype=torch.float64)
    B = torch.randn(n, n, dtype=torch.float64)
    t_loop = bench(lambda: matmul_loop(A.tolist(), B.tolist()), repeat=1)
    t_blas = bench(lambda: A @ B, repeat=20)
    print(f"n={n}: for 文 {t_loop * 1e3:.1f} ms ({gflops(n, t_loop):.3f} GFLOP/s)"
          f"  torch {t_blas * 1e3:.3f} ms ({gflops(n, t_blas):.1f} GFLOP/s)"
          f"  {t_loop / t_blas:,.0f} 倍")

print("== 行列サイズ・スレッド数・型 ==")
rows = []
for threads in (1, 2, 4):
    torch.set_num_threads(threads)
    for dtype in (torch.float32, torch.float64):
        for n in (32, 64, 128, 256, 512, 1024, 2048):
            A = torch.randn(n, n, dtype=dtype)
            B = torch.randn(n, n, dtype=dtype)
            rep = 3 if n >= 1024 else 7
            t = bench(lambda: A @ B, repeat=rep)
            rows.append((threads, str(dtype)[6:], n, gflops(n, t)))
print("スレッド 型       n=64    n=512   n=2048  (GFLOP/s)")
for th in (1, 2, 4):
    for dt in ("float32", "float64"):
        v = {r[2]: r[3] for r in rows if r[0] == th and r[1] == dt}
        print(f"{th:8d} {dt:7s} {v[64]:6.1f} {v[512]:7.1f} {v[2048]:7.1f}")
with open("ch12/out/matmul_bench.csv", "w") as f:
    f.write("threads,dtype,n,gflops\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.2f}\n")

# [snippet: batched]
torch.set_num_threads(2)
X = torch.randn(256, 64, 64)                 # 64×64 の行列が 256 個
W = torch.randn(256, 64, 64)
def one_by_one():
    return [X[i] @ W[i] for i in range(256)]  # 1 個ずつ 256 回呼ぶ
def batched():
    return torch.bmm(X, W)                    # まとめて 1 回で呼ぶ
t1, t2 = bench(one_by_one), bench(batched)
# [end: batched]
print("== 小さな行列積 256 個（2 スレッド）==")
print(f"1 個ずつ {t1 * 1e3:.2f} ms, まとめて {t2 * 1e3:.2f} ms, {t1 / t2:.1f} 倍")

# 図用: 行 = n、列 = 条件 の横長の CSV
with open("ch12/out/matmul_wide.csv", "w") as f:
    f.write("n,f32t1,f32t2,f32t4,f64t4\n")
    for n in (32, 64, 128, 256, 512, 1024, 2048):
        v = {(r[0], r[1]): r[3] for r in rows if r[2] == n}
        f.write(f"{n},{v[(1, 'float32')]:.2f},{v[(2, 'float32')]:.2f},"
                f"{v[(4, 'float32')]:.2f},{v[(4, 'float64')]:.2f}\n")
