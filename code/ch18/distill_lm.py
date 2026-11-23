"""第 18 章: 漱石の次トークン予測で、教師 base.pt の確率を真似る効果を測る。

python ch18/distill_lm.py {hard,soft} [frac] [steps] [tau] [alpha]
  hard: 正解だけ（交差エントロピー） / soft: 式 (eq:distill-loss) で蒸留
  frac: 漱石の学習用トークン列の先頭から使う割合（例 0.03）
"""
import csv
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from llmbook.corpus import DATA_DIR
from llmbook.model import GPT, GPTConfig
from llmbook.train import get_batch, load_checkpoint, lr_at, make_optimizer

from distill import distill_loss

torch.set_num_threads(2)
cond = sys.argv[1]
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
tau = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
alpha = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
OUT = DATA_DIR.parent / "ch18" / "out"


# [snippet: lm-data]
def soseki(name: str) -> torch.Tensor:
    """train.bin / val.bin のうち、最初の <|endoftext|> までの漱石の部分。"""
    a = np.fromfile(DATA_DIR / f"{name}.bin", dtype=np.uint16)
    eot = int(np.argmax(a == 4088))   # 漱石の後に区切り、その後に計算ドリル
    return torch.from_numpy(a[:eot].astype(np.int64))


train_all, val = soseki("train"), soseki("val")
train = train_all[: int(len(train_all) * frac)]   # 先頭の一部だけ使う
B, T = 32, 128
g = torch.Generator().manual_seed(123)        # 全条件で同じ検証バッチ
idx = torch.randint(len(val) - T - 1, (20, B), generator=g)
val_batches = [(torch.stack([val[i:i + T] for i in row]),
                torch.stack([val[i + 1:i + T + 1] for i in row]))
               for row in idx]
# [end: lm-data]

teacher = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt").eval()
for p in teacher.parameters():
    p.requires_grad_(False)
torch.manual_seed(0)
student = GPT(GPTConfig(vocab_size=4096, block_size=128, n_layer=2,
                        n_head=4, n_embd=96))


@torch.no_grad()
def val_loss(model) -> float:
    model.eval()
    ls = [F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten()).item()
          for x, y in val_batches]
    model.train()
    return sum(ls) / len(ls)


tag = f"{cond}_f{frac:g}" + (f"_t{tau:g}_a{alpha:g}" if cond == "soft" else "")
print(f"cond={cond} frac={frac} train_tokens={len(train):,} "
      f"val_tokens={len(val):,} steps={steps} tau={tau} alpha={alpha}")
print(f"teacher val loss: {val_loss(teacher):.4f}", flush=True)
opt = make_optimizer(student, 1e-2)
torch.manual_seed(1)
log, t0 = [], time.time()
# [snippet: lm-loop]
for step in range(steps + 1):
    if step % 50 == 0:
        log.append((step, val_loss(student), time.time() - t0))
        print(f"step {step:5d} val {log[-1][1]:.4f} "
              f"({log[-1][2]:.0f}s)", flush=True)
    if step == steps:
        break
    x, y = get_batch(train, B, T)
    z_s = student(x)[0]
    if cond == "soft":
        with torch.no_grad():
            z_t = teacher(x)[0]            # 教師は前向き計算だけ
        loss, _, _ = distill_loss(z_s, z_t, y, tau, alpha)
    else:
        loss = F.cross_entropy(z_s.flatten(0, 1), y.flatten())
    for gr in opt.param_groups:
        gr["lr"] = lr_at(step, 1e-2, 30, steps)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
# [end: lm-loop]
best = min(log, key=lambda r: r[1])
print(f"best val {best[1]:.4f} at step {best[0]}; final val {log[-1][1]:.4f}; "
      f"time {time.time() - t0:.0f}s")
with open(OUT / f"distill_lm_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "val_loss", "time"])
    w.writerows(log)
