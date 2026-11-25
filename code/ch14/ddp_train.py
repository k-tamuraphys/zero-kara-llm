"""torchrun で起動するデータ並列の学習（GPU クラスタでもこの形で使う）。

1 台・CPU: torchrun --standalone --nproc_per_node=2 ch14/ddp_train.py
複数ノード（GPU）     : train.slurm を参照
"""

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from llmbook.train import get_batch, make_optimizer
from ch14.common import load_bin, tiny_gpt

# [snippet: init]
rank = int(os.environ["RANK"])              # 全体での通し番号
local_rank = int(os.environ["LOCAL_RANK"])  # このノードの中での番号
world = int(os.environ["WORLD_SIZE"])       # プロセスの総数
if torch.cuda.is_available():
    device = f"cuda:{local_rank}"            # 1 プロセス = 1 GPU
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")          # GPU どうしは NCCL
else:
    device = "cpu"
    torch.set_num_threads(1)
    dist.init_process_group("gloo")          # CPU では gloo
# [end: init]

# [snippet: loop]
B_LOCAL, ACCUM = 4, 2                     # 全体のバッチ = 4×2×world
STEPS = int(os.environ.get("STEPS", 30))     # 最後のステップ番号
CKPT = "checkpoints/ch14_ddp.pt"
data = load_bin("train")
model = DDP(tiny_gpt(seed=0).to(device))     # 重みはランク 0 から配られる
torch.manual_seed(1234 + rank)               # データの引き方はランクごと
opt = make_optimizer(model, lr=1e-3)
start = 0
if os.path.exists(CKPT):                     # 途中から再開できるようにする
    ck = torch.load(CKPT, map_location=device)
    model.module.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start = ck["step"] + 1
    if rank == 0:
        print(f"{CKPT} の step {start} から再開します")
t0 = time.perf_counter()
for step in range(start, STEPS):
    for micro in range(ACCUM):               # 勾配累積
        x, y = get_batch(data, B_LOCAL, 64, device)
        sync = micro == ACCUM - 1            # 最後のマイクロバッチだけ通信
        ctx = model.no_sync() if not sync else torch.enable_grad()
        with ctx:
            _, loss = model(x, y)
            (loss / ACCUM).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 10 == 9 and rank == 0:         # 保存と表示はランク 0 だけ
        torch.save({"model": model.module.state_dict(),
                    "opt": opt.state_dict(), "step": step}, CKPT)
        print(f"step {step+1}: loss {loss.item():.3f}"
              f"（ランク 0 の値） 保存しました")
# [end: loop]
dist.barrier()
if rank == 0:
    tok = (STEPS - start) * B_LOCAL * ACCUM * world * 64
    print(f"world={world}: {tok} トークンを {time.perf_counter()-t0:.1f} 秒")
dist.destroy_process_group()
