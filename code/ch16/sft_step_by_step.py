"""計算アシスタントの SFT を一段ずつ作り直す（scripts/sft_chat.py と同設定）。

学習の途中で、損失・対話の正答率・漱石の損失・ドリル形式の正答率を記録する。
実行:  python ch16/sft_step_by_step.py
出力:  ch16/out/sft_loss.csv, sft_eval.csv, sft_step_by_step.txt,
       checkpoints/ch16_sft.pt          CPU で約 4 分。
"""

import csv
import random
import time

import torch

from common import CKPT_DIR, OUT_DIR, device, load_base, soseki_val_loss, tok
from llmbook.arith import drill_accuracy
from llmbook.chat import (chat_accuracy, chat_reply, collate,
                          encode_chat_example, make_chat_example)
from llmbook.train import lr_at, make_optimizer, save_checkpoint

torch.manual_seed(0)
model = load_base()

# [snippet: sft-data]
# 学習データ: 対話 3,000 件（評価用の問題とは乱数の種が異なる）
rng = random.Random(0)
examples = [encode_chat_example(tok, make_chat_example(rng)[0])
            for _ in range(3000)]
pad_id = tok.special_tokens["<|pad|>"]
# [end: sft-data]


def evaluate(step: int) -> dict:
    """対話の正答率と、事前学習で得た能力（漱石・ドリル形式）を測る。"""
    return {"step": step,
            "chat_acc": chat_accuracy(model, tok, device=device),
            "soseki_loss": soseki_val_loss(model),
            "drill_acc": drill_accuracy(model, tok, device=device)}


evals, losses = [], []
t0 = time.time()
# [snippet: sft-loop]
max_steps, batch_size, max_lr = 600, 32, 3e-4
optimizer = make_optimizer(model, max_lr, weight_decay=0.0)
for step in range(max_steps + 1):
    if step % 50 == 0:
        evals.append(evaluate(step))  # 途中経過を測る（学習には影響しない）
    if step == max_steps:
        break
    for group in optimizer.param_groups:  # ウォームアップ + コサイン減衰
        group["lr"] = lr_at(step, max_lr, warmup_steps=30,
                            max_steps=max_steps)
    batch = rng.sample(examples, batch_size)  # 32 件を無作為に選ぶ
    x, y = collate(batch, pad_id)  # (B, T) の入力と正解
    model.train()
    _, loss = model(x.to(device), y.to(device))  # 正解のある位置だけの平均
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append({"step": step, "loss": loss.item()})
# [end: sft-loop]
    if step % 50 == 0:
        print(evals[-1], losses[-1], f"{time.time() - t0:.0f} 秒", flush=True)

save_checkpoint(model, CKPT_DIR / "ch16_sft.pt")
for fname, rows in [("sft_loss.csv", losses), ("sft_eval.csv", evals)]:
    with open(OUT_DIR / fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

with open(OUT_DIR / "sft_step_by_step.txt", "w") as f:
    for r in losses:
        if r["step"] in (0, 100, 300, 599):
            print(f"step {r['step']:3d} | 損失 {r['loss']:.3f}", file=f)
    for r in evals:
        if r["step"] in (0, 100, 300, 600):
            acc = r["chat_acc"]
            print(f"{r['step']:3d} 回更新後 | 対話の正答率 {acc:.1%}",
                  file=f)
    for q in ["12+35は？", "300から123を引くといくつですか？", "7足す8は？",
              "456と789を足すといくつですか？"]:
        print(q, "->", repr(chat_reply(model, tok, q, device=device)),
              file=f)
print(f"所要時間 {time.time() - t0:.0f} 秒")
