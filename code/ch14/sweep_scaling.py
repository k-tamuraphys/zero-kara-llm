"""1 プロセスあたりの仕事量を変えて、スケーリング効率がどう変わるかを測る。

comm_bench.py の学習の測り方（パラメータごとに all-reduce）はそのままに、
1 プロセスあたりの系列数 B_LOCAL を 2, 8, 32 と変える。
実行: python -m ch14.sweep_scaling   → out/sweep_scaling.csv
"""

import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch14.common import fixed_batch, load_bin, setup, tiny_gpt

B_LIST = [2, 8, 32]       # 1 プロセスあたりの系列数（計算の量）
W_LIST = [1, 2, 4]        # プロセス数
STEPS, REPS = 8, 3        # 測るステップ数と、繰り返しの回数


def bench(rank: int, W: int, B: int, data) -> tuple[float, float]:
    """1 ステップあたりの計算と通信の時間（秒）を返す。"""
    model = tiny_gpt()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t_comp = t_comm = 0.0
    for step in range(STEPS + 2):
        x, y = fixed_batch(data, step * W + rank, B)
        t0 = time.perf_counter()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        t1 = time.perf_counter()
        for p in model.parameters():          # comm_bench と同じ同期の仕方
            dist.all_reduce(p.grad)
            p.grad /= W
        t2 = time.perf_counter()
        opt.step()
        if step >= 2:                         # 最初の 2 回は準備なので捨てる
            t_comp += (t1 - t0) + (time.perf_counter() - t2)
            t_comm += t2 - t1
    return t_comp / STEPS, t_comm / STEPS


def worker(rank, W, B, q):
    setup(rank, W, port=29700 + 10 * W + B_LIST.index(B))
    data = load_bin("train")
    comp, comm = bench(rank, W, B, data)
    t = torch.tensor([comp, comm])
    dist.all_reduce(t)                        # 全ランクの平均を報告する
    if rank == 0:
        q.put((t / W).tolist())
    dist.destroy_process_group()


def run(W: int, B: int) -> list[float]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, W, B, q)) for r in range(W)]
    for p in ps:
        p.start()
    res = q.get()
    for p in ps:
        p.join()
    return res


if __name__ == "__main__":
    rows = []
    for rep in range(REPS):                   # 混雑でぶれるので繰り返す
        for B in B_LIST:
            for W in W_LIST:
                comp, comm = run(W, B)
                tok = W * B * 64 / (comp + comm)   # 全体で 1 秒に何トークン
                rows.append((rep, B, W, comp, comm, tok))
                print(f"rep {rep} B={B:2d} W={W}: 計算 {comp*1e3:7.1f} ms"
                      f"  通信 {comm*1e3:6.1f} ms  {tok:8.0f} トークン/秒")
    # 効率 = W プロセスの速さ ÷（1 プロセスの速さ × W）。同じ rep・B で比べる
    base = {(r[0], r[1]): r[5] for r in rows if r[2] == 1}
    with open("ch14/out/sweep_scaling.csv", "w") as f:
        f.write("rep,B,W,comp_ms,comm_ms,tok_per_s,eff\n")
        for rep, B, W, comp, comm, tok in rows:
            eff = tok / (W * base[(rep, B)])
            f.write(f"{rep},{B},{W},{comp*1e3:.2f},{comm*1e3:.2f},"
                    f"{tok:.0f},{eff:.3f}\n")
    # 図に使う代表値: 3 回の効率の中央値
    with open("ch14/out/sweep_scaling_median.csv", "w") as f:
        f.write("B,W,eff,comm_share\n")
        for B in B_LIST:
            for W in W_LIST:
                rs = [r for r in rows if r[1] == B and r[2] == W]
                effs = sorted(r[5] / (W * base[(r[0], B)]) for r in rs)
                shares = sorted(r[4] / (r[3] + r[4]) for r in rs)
                f.write(f"{B},{W},{effs[len(effs)//2]:.3f},"
                        f"{shares[len(shares)//2]:.3f}\n")
