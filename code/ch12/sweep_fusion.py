"""torch.compile（演算の融合）の効果を、CPU で何回か測って幅を見る。

4 種類の要素ごとの計算を、そのまま（eager）と torch.compile 後で比べる。
1 回の測定（bench: 10 回の中央値）を 5 回くり返し、最小・中央・最大を出す。
"""
import time
import torch

torch.manual_seed(0)
torch.set_num_threads(2)            # 本章のほかの測定と同じ 2 スレッド


def bench(fn, repeat=10):
    fn()                            # 準備運動の 1 回は捨てる
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def mul2(x):                        # 読んで 1 回掛けて書くだけ
    return x * 2.0


def poly(x):                        # GELU の式から tanh を除いた 6 回の演算
    return 0.5 * x * (1 + 0.79788456 * (x + 0.044715 * x**3))


def tanh1(x):                       # tanh だけ（1 回の演算）
    return torch.tanh(x)


def gelu_chain(x):                  # speedup.py と同じ 7 回の演算
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x**3)))


x = torch.randn(4096, 4096)         # 1,677 万要素 = 64 MiB
funcs = {"mul2": mul2, "poly": poly, "tanh": tanh1, "gelu": gelu_chain}
compiled = {}
for name, f in funcs.items():
    c = torch.compile(f)
    t0 = time.perf_counter()
    c(x)                            # 初回の呼び出しで融合したプログラムを作る
    print(f"{name:5s}: 初回（変換を含む） {time.perf_counter() - t0:.1f} s")
    compiled[name] = c
    diff = (c(x) - f(x)).abs().max().item()
    print(f"{name:5s}: eager との最大差 {diff:.1e}")

ROUNDS = 5
rows = []
for name in funcs:
    for mode, fn in (("eager", funcs[name]), ("compile", compiled[name])):
        ts = sorted(bench(lambda: fn(x)) * 1e3 for _ in range(ROUNDS))
        rows.append((name, mode, ts[0], ts[ROUNDS // 2], ts[-1]))
        print(f"{name:5s} {mode:7s}: 中央 {ts[ROUNDS // 2]:6.1f} ms"
              f"（最小 {ts[0]:6.1f}, 最大 {ts[-1]:6.1f}）")

with open("ch12/out/sweep_fusion.csv", "w") as f:
    f.write("name,mode,tmin,tmed,tmax\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f}\n")

# 図に使う横長の表: 関数ごとに 1 行。lo・hi は中央値から最小・最大までの幅
with open("ch12/out/sweep_fusion_wide.csv", "w") as f:
    f.write("name,e_med,e_lo,e_hi,c_med,c_lo,c_hi\n")
    for i in range(0, len(rows), 2):
        e, c = rows[i], rows[i + 1]           # eager の行と compile の行
        f.write(f"{e[0]},{e[3]:.2f},{e[3] - e[2]:.2f},{e[4] - e[3]:.2f},"
                f"{c[3]:.2f},{c[3] - c[2]:.2f},{c[4] - c[3]:.2f}\n")
