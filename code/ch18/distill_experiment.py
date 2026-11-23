"""第 18 章: 教師 sft.pt から小さな生徒（2 層・96 次元）へ能力を移す実験。

python ch18/distill_experiment.py {hard,soft,seq} [N] [steps]
  hard: 正解だけで学習 / soft: 教師の分布で蒸留 / seq: 教師の生成文で SFT
"""
import csv
import random
import sys
import time

import torch

from llmbook.chat import (IGNORE, chat_accuracy, collate, encode_chat_example,
                          make_chat_example)
from llmbook.corpus import DATA_DIR
from llmbook.model import GPT, GPTConfig
from llmbook.tokenizer import BPETokenizer
from llmbook.train import (load_checkpoint, lr_at, make_optimizer,
                           save_checkpoint)

from distill import distill_loss, teacher_answers

torch.set_num_threads(2)
cond = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
MV = int(sys.argv[4]) if len(sys.argv) > 4 else 999   # 数の最大値
CK = DATA_DIR.parent / "checkpoints"
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
pad = tok.special_tokens["<|pad|>"]
teacher = load_checkpoint(CK / "sft.pt").eval()
for p in teacher.parameters():
    p.requires_grad_(False)

# [snippet: make-data]
rng = random.Random(0)
data = [make_chat_example(rng, MV)[0] for _ in range(N)]  # 第 16 章と同じ
if cond == "seq":                     # 正解を教師の答えに差し替える
    qs = [m[0]["content"] for m in data]
    for m, a in zip(data, teacher_answers(teacher, tok, qs)):
        m[1]["content"] = a
examples = [encode_chat_example(tok, m) for m in data]
# [end: make-data]
if cond == "seq":   # 教師の答えのうち正解だった割合を数える
    rng = random.Random(0)
    gold = [make_chat_example(rng, MV)[0][1]["content"] for _ in range(N)]
    n_ok = sum(m[1]["content"] == g for m, g in zip(data, gold))
    print(f"teacher answers correct: {n_ok}/{N}")

# [snippet: student]
torch.manual_seed(0)
student = GPT(GPTConfig(vocab_size=4096, block_size=128, n_layer=2,
                        n_head=4, n_embd=96))
# [end: student]
print(f"cond={cond} N={N} steps={steps} max_value={MV} "
      f"student={student.num_params():,} teacher={teacher.num_params():,}")
opt = make_optimizer(student, 1e-3)
log = []
t0 = time.time()
g = torch.Generator().manual_seed(1)
# [snippet: train-loop]
for step in range(steps):
    for grp in opt.param_groups:
        grp["lr"] = lr_at(step, 1e-3, 100, steps)
    pick = torch.randint(len(examples), (32,), generator=g).tolist()
    x, y = collate([examples[i] for i in pick], pad)
    z_s, _ = student(x)                                  # (B, T, V)
    if cond == "soft":
        with torch.no_grad():
            z_t, _ = teacher(x)                          # 教師は固定
        loss, kl, ce = distill_loss(z_s, z_t, y, tau=2.0, alpha=0.5)
    else:                                  # hard と seq は普通の SFT
        loss = torch.nn.functional.cross_entropy(
            z_s.flatten(0, 1), y.flatten(), ignore_index=IGNORE)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
# [end: train-loop]
    if (step + 1) % 300 == 0:
        acc = chat_accuracy(student, tok, 200, seed=4242, max_value=MV)
        student.train()
        log.append(dict(step=step + 1, loss=round(loss.item(), 4), acc=acc))
        print(f"step {step:5d} loss {loss.item():.4f} acc200 {acc:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
acc = chat_accuracy(student, tok, 1000, seed=777, max_value=MV)
print(f"final chat accuracy (1000, seed 777): {acc:.3f}  "
      f"time {time.time() - t0:.0f}s")
tag = f"{cond}_n{N}" + (f"_mv{MV}" if MV != 999 else "")
with open(f"ch18/out/distill_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step", "loss", "acc"])
    w.writeheader()
    w.writerows(log)
save_checkpoint(student, CK / f"ch18_{tag}.pt", final_acc=acc)
