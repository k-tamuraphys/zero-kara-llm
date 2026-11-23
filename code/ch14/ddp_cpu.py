"""CPU の複数プロセスでデータ並列を動かし、1 プロセスの大バッチ学習と比べる。

実行: python -m ch14.ddp_cpu 4   （引数はプロセス数。2〜4）
"""

import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ch14.common import fixed_batch, flat_params, load_bin, setup, tiny_gpt

B_GLOBAL, STEPS = 16, 20          # 全体のバッチ 16 件、20 ステップ


def single_process() -> tuple[list[float], torch.Tensor]:
    """比較の基準: 1 プロセスが全体のバッチ 16 件をまとめて学習する。"""
    torch.set_num_threads(1)
    data = load_bin("train")
    model = tiny_gpt(seed=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(STEPS):
        x, y = fixed_batch(data, step, B_GLOBAL)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, flat_params(model)


# [snippet: manual]
def train_manual(model, opt, data, rank, W):
    """all-reduce を自分で呼ぶ、いちばん素朴なデータ並列。"""
    losses = []
    for step in range(STEPS):
        x, y = fixed_batch(data, step, B_GLOBAL)       # 全体 (16, 64)
        x, y = x.chunk(W)[rank], y.chunk(W)[rank]      # 自分の分 (16/W, 64)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()                                # 自分の分の勾配
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)  # 全員の和
            p.grad /= W                                # 和 → 平均
        opt.step()                                     # 全員が同じ更新
        losses.append(loss.item())
    return losses
# [end: manual]


def worker(rank: int, W: int, mode: str, q) -> None:
    setup(rank, W)
    data = load_bin("train")
    # [snippet: ddp]
    model = tiny_gpt(seed=rank)          # わざとランクごとに違う初期値
    before = flat_params(model)[:3].tolist()
    if mode == "ddp":
        model = DDP(model)               # 作った瞬間にランク 0 の重みを配る
    after = flat_params(model)[:3].tolist()
    # [end: ddp]
    if mode == "manual":                 # 素朴版はランク 0 の重みを自分で配る
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        after = flat_params(model)[:3].tolist()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if mode == "manual":
        losses = train_manual(model, opt, data, rank, W)
    else:
        losses = []
        for step in range(STEPS):
            x, y = fixed_batch(data, step, B_GLOBAL)
            x, y = x.chunk(W)[rank], y.chunk(W)[rank]
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()              # DDP はここで勾配を平均する
            opt.step()
            losses.append(loss.item())
    # 各ランクの損失（自分の分の平均）を集め、全体の平均損失に直す
    t = torch.tensor(losses)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    q.put((rank, before, after, (t / W).tolist(), flat_params(model).numpy()))
    dist.destroy_process_group()


def run(W: int, mode: str):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, W, mode, q))
             for r in range(W)]
    for p in procs:
        p.start()
    res = sorted([q.get() for _ in range(W)], key=lambda r: r[0])
    for p in procs:
        p.join()
    return [(r, b, a, l, torch.from_numpy(p)) for r, b, a, l, p in res]


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ref_losses, ref_params = single_process()
    print(f"プロセス数 W = {W}、全体のバッチ {B_GLOBAL} 件"
          f"（1 プロセスあたり {B_GLOBAL // W} 件）")
    for mode in ["manual", "ddp"]:
        res = run(W, mode)
        print(f"--- {mode} ---")
        for rank, before, after, _, _ in res:
            print(f"rank {rank}: 重みの先頭 {before[0]:+.4f} → {after[0]:+.4f}")
        losses = res[0][3]
        for s in [0, 1, 9, 19]:
            print(f"step {s:2d}: 損失 DP {losses[s]:.6f}"
                  f"  1 プロセス {ref_losses[s]:.6f}")
        p0 = res[0][4]
        spread = max((r[4] - p0).abs().max().item() for r in res)
        print(f"ランク間の重みの最大差     : {spread:.1e}")
        print(f"1 プロセスとの重みの最大差 : {(p0 - ref_params).abs().max():.1e}")
