"""誤差逆伝播法の手間（時間）とメモリを、MLP で測る。

(1) 隠れ層の幅を 5 通りに変え、順伝播と逆伝播の時間を測る
    （データ 1,024 件、隠れ層 4 枚で固定）。
(2) データの件数 N を 5 通りに変え、逆伝播のために残される途中の値の
    バイト数を数える（幅 256、隠れ層 4 枚で固定）。
    パラメータのバイト数と、推論（no_grad）のときに残る量も並べる。
共有 CPU なので、時間は各 7 回測って中央値をとる。
出力: ch04/out/sweep_backprop_time.csv、ch04/out/sweep_backprop_mem.csv
"""

import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(2)
DEPTH = 4  # 隠れ層の枚数


def make_mlp(width):
    """入力 2、幅 width の隠れ層 DEPTH 枚（ReLU）、出力 3 の MLP。"""
    sizes = [2] + [width] * DEPTH
    layers = []
    for i in range(DEPTH):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(width, 3))


def saved_bytes(model, X, Y, grad=True):
    """順伝播で、逆伝播のために残された途中の値のバイト数を数える。"""
    params = {p.data_ptr() for p in model.parameters()}
    seen = {}

    def pack(t):  # 逆伝播のために値が残されるたびに呼ばれる
        if t.data_ptr() not in params:  # パラメータそのものは数えない
            seen[t.data_ptr()] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.set_grad_enabled(grad):
            F.cross_entropy(model(X), Y)
    return sum(seen.values())


def median_ms(fn, reps=8):
    """fn を reps 回実行し、1 回目（準備が入る）を除いた中央値 [ミリ秒]。"""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts[1:]) * 1000


# --- (1) 幅を変えて、時間を測る
print("(1) データ 1,024 件、隠れ層 4 枚。時間は中央値")
N = 1024
X, Y = torch.randn(N, 2), torch.randint(0, 3, (N,))
with open("ch04/out/sweep_backprop_time.csv", "w") as f:
    f.write("width,params,fwd_ms,bwd_ms,ratio\n")
    for width in (64, 128, 256, 512, 1024):
        model = make_mlp(width)
        P = sum(p.numel() for p in model.parameters())
        fwd, bwd = [], []
        for _ in range(8):
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(X), Y)  # 順伝播
            t1 = time.perf_counter()
            model.zero_grad()
            loss.backward()  # 逆伝播
            t2 = time.perf_counter()
            fwd.append(t1 - t0)
            bwd.append(t2 - t1)
        fm = statistics.median(fwd[1:]) * 1000
        bm = statistics.median(bwd[1:]) * 1000
        f.write(f"{width},{P},{fm:.3f},{bm:.3f},{bm / fm:.2f}\n")
        print(f"  幅 {width:4d}  パラメータ {P:9,d}  順伝播 {fm:7.2f} ミリ秒"
              f"  逆伝播 {bm:7.2f} ミリ秒（{bm / fm:.1f} 倍）")

# --- (2) データの件数を変えて、残される途中の値を数える
print("(2) 幅 256、隠れ層 4 枚。単位は MB（100 万バイト）")
model = make_mlp(256)
p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
with open("ch04/out/sweep_backprop_mem.csv", "w") as f:
    f.write("N,saved_mb,param_mb,nograd_mb,fwd_ms,fwd_nograd_ms\n")
    for N in (128, 256, 512, 1024, 2048):
        X, Y = torch.randn(N, 2), torch.randint(0, 3, (N,))
        s = saved_bytes(model, X, Y) / 1e6
        s0 = saved_bytes(model, X, Y, grad=False) / 1e6
        ft = median_ms(lambda: F.cross_entropy(model(X), Y))
        with torch.no_grad():
            f0 = median_ms(lambda: F.cross_entropy(model(X), Y))
        f.write(f"{N},{s:.4f},{p_bytes / 1e6:.4f},{s0:.4f},"
                f"{ft:.3f},{f0:.3f}\n")
        print(f"  N {N:5d}  残される途中の値 {s:6.2f}  パラメータ "
              f"{p_bytes / 1e6:.2f}  推論のとき {s0:.2f}")
