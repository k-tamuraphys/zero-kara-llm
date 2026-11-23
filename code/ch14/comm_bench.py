"""通信と計算の時間を測る: all-reduce の速さと、プロセス数を変えた学習の速さ。

実行: python -m ch14.comm_bench   → out/allreduce.csv, out/scaling.csv
"""

import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch14.common import fixed_batch, load_bin, setup, tiny_gpt

SIZES = [10**3, 10**4, 10**5, 10**6, 10**7]   # 送る float32 の個数
B_LOCAL, STEPS = 8, 8


def bench_allreduce(rank: int):
    out = []
    for n in SIZES:
        t = torch.ones(n)
        times = []
        for _ in range(7):
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_reduce(t)
            times.append(time.perf_counter() - t0)
        out.append(sorted(times)[3])            # 中央値
    return out


# [snippet: timed-step]
def bench_train(rank: int, W: int, data):
    model = tiny_gpt()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t_comp = t_comm = 0.0
    for step in range(STEPS + 2):
        x, y = fixed_batch(data, step * W + rank, B_LOCAL)  # 1 人 8 件
        t0 = time.perf_counter()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        t1 = time.perf_counter()
        for p in model.parameters():                     # 勾配の同期
            dist.all_reduce(p.grad)
            p.grad /= W
        t2 = time.perf_counter()
        opt.step()
        if step >= 2:                                    # 最初の 2 回は捨てる
            t_comp += (t1 - t0) + (time.perf_counter() - t2)
            t_comm += t2 - t1
    return t_comp / STEPS, t_comm / STEPS
# [end: timed-step]


def worker(rank, W, q):
    setup(rank, W, port=29600 + W)
    data = load_bin("train")
    ar = bench_allreduce(rank) if W > 1 else [0.0] * len(SIZES)
    comp, comm = bench_train(rank, W, data)
    t = torch.tensor([comp, comm])
    dist.all_reduce(t)
    if rank == 0:
        q.put((ar, (t / W).tolist()))
    dist.destroy_process_group()


def run(W):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, W, q)) for r in range(W)]
    for p in ps:
        p.start()
    res = q.get()
    for p in ps:
        p.join()
    return res


if __name__ == "__main__":
    rows_ar, rows_sc = [], []
    for rep in range(3):                       # 混雑でぶれるので 3 回測る
        for W in [1, 2, 4]:
            ar, (comp, comm) = run(W)
            step = comp + comm
            tok = W * B_LOCAL * 64 / step       # 全体で 1 秒に何トークン
            rows_sc.append((rep, W, comp, comm, tok))
            if W > 1:
                rows_ar += [(rep, W, n, s) for n, s in zip(SIZES, ar)]
            print(f"rep {rep} W={W}: 計算 {comp*1e3:6.1f} ms"
                  f"  通信 {comm*1e3:6.1f} ms  {tok:8.0f} トークン/秒")
    with open("ch14/out/scaling.csv", "w") as f:
        f.write("rep,W,comp_ms,comm_ms,tok_per_s\n")
        for r in rows_sc:
            f.write(f"{r[0]},{r[1]},{r[2]*1e3:.2f},{r[3]*1e3:.2f},{r[4]:.0f}\n")
    with open("ch14/out/allreduce.csv", "w") as f:
        f.write("rep,W,n,mbytes,ms,gbps\n")
        for rep, W, n, s in rows_ar:
            mb = n * 4 / 1e6
            f.write(f"{rep},{W},{n},{mb:.3f},{s*1e3:.3f},{mb/1e3/s:.3f}\n")
