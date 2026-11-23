"""リング all-reduce を send/recv だけで自作し、dist.all_reduce と比べる。

実行: python -m ch14.ring_allreduce   （4 プロセス）
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch14.common import setup

W = 4


# [snippet: ring]
def ring_all_reduce(t: torch.Tensor, rank: int, W: int, log=None):
    """t を W 個の塊に分け、隣へ渡すだけで全員の和を作る。"""
    chunks = list(t.chunk(W))                 # 塊 0..W-1（t と記憶を共有）
    right, left = (rank + 1) % W, (rank - 1) % W
    buf = torch.empty_like(chunks[0])
    # 前半 reduce-scatter: W-1 回。受け取った塊を自分の塊に足す
    for s in range(W - 1):
        send_i, recv_i = (rank - s) % W, (rank - s - 1) % W
        req = dist.isend(chunks[send_i].clone(), right)
        dist.recv(buf, left)
        req.wait()
        chunks[recv_i] += buf
        if log is not None:
            log.append(t.tolist())
    # 後半 all-gather: W-1 回。完成した塊を右へ回し、上書きする
    for s in range(W - 1):
        send_i, recv_i = (rank - s + 1) % W, (rank - s) % W
        req = dist.isend(chunks[send_i].clone(), right)
        dist.recv(buf, left)
        req.wait()
        chunks[recv_i].copy_(buf)
        if log is not None:
            log.append(t.tolist())
    return t
# [end: ring]


def worker(rank, q):
    setup(rank, W, port=29700)
    # ランク r は塊 c に値 10^c × (r+1) を持つ（誰の値が混ざったか読める）
    t = torch.tensor([(rank + 1) * 10.0 ** c for c in range(W)])
    log = [t.tolist()]
    ring_all_reduce(t, rank, W, log)
    ref = torch.tensor([(rank + 1) * 10.0 ** c for c in range(W)])
    dist.all_reduce(ref)
    big = torch.randn(1000, generator=torch.Generator().manual_seed(rank))
    big_ref = big.clone()
    ring_all_reduce(big, rank, W)
    dist.all_reduce(big_ref)
    q.put((rank, log, (t - ref).abs().max().item(),
           (big - big_ref).abs().max().item()))
    dist.destroy_process_group()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, q)) for r in range(W)]
    for p in ps:
        p.start()
    res = sorted([q.get() for _ in range(W)])
    for p in ps:
        p.join()
    names = ["最初"] + [f"RS{s+1}" for s in range(W - 1)] + \
            [f"AG{s+1}" for s in range(W - 1)]
    print("     " + " | ".join(f"ランク {r}".ljust(14) for r in range(W)))
    for i, name in enumerate(names):
        row = " | ".join(",".join(f"{v:g}" for v in r[1][i]).ljust(17)
                         for r in res)
        print(f"{name:4s} {row}")
    for rank, _, d1, d2 in res:
        print(f"rank {rank}: dist.all_reduce との差 {d1:.0e}（4 要素）,"
              f" {d2:.1e}（1000 要素）")
