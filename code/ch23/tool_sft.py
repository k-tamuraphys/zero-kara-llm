"""sft.pt に電卓ツールの使い方を SFT で教える → checkpoints/ch23_tool_sft.pt

実行:  python ch23/tool_sft.py   出力: ch23/out/tool_sft_loss.csv,
       tool_sft_eval.csv, tool_sft.txt                      CPU で約 9 分。
"""

import csv
import random
import sys
import time

import torch

from common import CKPT_DIR, OUT_DIR, device, pad_id, tok
from mini_agent import MiniGPTLLM, agent_accuracy
from tools import make_tool_example
from llmbook.chat import collate, encode_chat_example
from llmbook.train import (load_checkpoint, lr_at, make_optimizer,
                           save_checkpoint)

max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
torch.manual_seed(0)
model = load_checkpoint(CKPT_DIR / "sft.pt", device)  # 第 16 章の出発点

# [snippet: tool-sft-data]
rng = random.Random(0)
kinds = ["add", "mul", "two"]  # 自分で答える / 電卓 / 電卓（2 演算）
examples = [encode_chat_example(tok, make_tool_example(rng, kinds[i % 3])[0],
                                learn_role=True)  # 役割トークンも学ぶ
            for i in range(4500)]
# [end: tool-sft-data]

llm = MiniGPTLLM(model)
evals, losses = [], []
t0 = time.time()
# [snippet: tool-sft-loop]
# [snippet: tool-sft-eval]
batch_size, max_lr = 32, 3e-4
optimizer = make_optimizer(model, max_lr, weight_decay=0.0)
for step in range(max_steps + 1):
    if step % 100 == 0:  # 途中経過: 3 種類の問題を 50 問ずつ
        evals.append({"step": step} | {
            k: agent_accuracy(llm, k, n_problems=50)["acc"] for k in kinds})
        model.train()
        # [end: tool-sft-eval]
    if step == max_steps:
        break
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step, max_lr, warmup_steps=30,
                            max_steps=max_steps)
    x, y = collate(rng.sample(examples, batch_size), pad_id)
    _, loss = model(x.to(device), y.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append({"step": step, "loss": loss.item()})
# [end: tool-sft-loop]
    if step % 100 == 0:
        print(evals[-1], losses[-1], f"{time.time() - t0:.0f} 秒", flush=True)

save_checkpoint(model, CKPT_DIR / "ch23_tool_sft.pt")
for fname, rows in [("tool_sft_loss.csv", losses),
                    ("tool_sft_eval.csv", evals)]:
    with open(OUT_DIR / fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
with open(OUT_DIR / "tool_sft.txt", "w") as f:
    for r in losses:
        if r["step"] in (0, 100, 300, max_steps - 1):
            print(f"step {r['step']:3d} | 損失 {r['loss']:.3f}", file=f)
    for r in evals:
        print(f"{r['step']:3d} 回更新後 | 加減算 {r['add']:.0%}"
              f"  掛け算 {r['mul']:.0%}  掛けて足す {r['two']:.0%}", file=f)
print(f"所要時間 {time.time() - t0:.0f} 秒")
