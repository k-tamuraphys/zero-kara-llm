"""FSDP2（fully_shard）で重みを 2 プロセスに分割し、各自が持つ量を数える。

実行: python -m ch14.fsdp_cpu   （2 プロセス、CPU・gloo）
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import fully_shard

from ch14.common import fixed_batch, load_bin, setup, tiny_gpt

W, STEPS = 2, 5


def local_numel(model) -> int:
    """このプロセスが実際に記憶している重みの要素数。"""
    n = 0
    for p in model.parameters():
        n += p.to_local().numel() if hasattr(p, "to_local") else p.numel()
    return n


def worker(rank, q):
    setup(rank, W, port=29800)
    data = load_bin("train")
    model = tiny_gpt(seed=0)
    full = local_numel(model)
    # [snippet: fsdp]
    for block in model.blocks:          # ブロックごとに 1 単位として分割
        fully_shard(block)
    fully_shard(model)                  # 残り（埋め込み・出力層）も分割
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # [end: fsdp]
    mine = local_numel(model)
    losses = []
    for step in range(STEPS):
        x, y = fixed_batch(data, step, 16)
        x, y = x.chunk(W)[rank], y.chunk(W)[rank]
        _, loss = model(x, y)           # 層ごとに all-gather してから計算
        opt.zero_grad()
        loss.backward()                 # 勾配は reduce-scatter で分割平均
        opt.step()                      # 自分の担当分だけ更新
        t = loss.detach().clone()
        dist.all_reduce(t)
        losses.append(t.item() / W)
    q.put((rank, full, mine, losses))
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
    for rank, full, mine, _ in res:
        print(f"rank {rank}: 重み {full:,} 個のうち {mine:,} 個を担当"
              f"（{mine / full:.1%}）")
    print("全体の平均損失:", " ".join(f"{v:.6f}" for v in res[0][3]))
