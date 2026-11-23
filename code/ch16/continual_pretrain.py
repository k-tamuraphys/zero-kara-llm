"""継続事前学習の小実験: 基盤モデルに新しい領域「掛け算ドリル」を読ませる。

実行:  python ch16/continual_pretrain.py high_lr  （ほかに low_lr, mix）
出力:  ch16/out/cpt_<設定>.csv, ch16/out/cpt_<設定>.txt,
       checkpoints/ch16_cpt_<設定>.pt
       3 設定を同時に実行して、それぞれ CPU で約 13 分。
"""

import csv
import sys
import time

import torch

from common import (CKPT_DIR, OUT_DIR, device, load_base, soseki_val_loss,
                    tok, train_tokens)
from llmbook.arith import complete, drill_accuracy, drill_text
from llmbook.train import get_batch, lr_at, make_optimizer, save_checkpoint

# [snippet: cpt-settings]
# 設定ごとの (最大学習率, 新領域の割合)。割合 0.5 は「半分は元のデータ」
SETTINGS = {"high_lr": (1e-3, 1.0), "low_lr": (1e-4, 1.0), "mix": (1e-3, 0.5)}
name = sys.argv[1] if len(sys.argv) > 1 else "high_lr"
max_lr, new_ratio = SETTINGS[name]
max_steps, batch_size, block_size = 300, 32, 128
# [end: cpt-settings]

torch.manual_seed(0)
# [snippet: cpt-data]
# 新しい領域: 2 桁までの掛け算ドリル（事前学習では一度も見ていない）
new_text = drill_text(60_000, seed=7, max_value=99, ops="*")
new_data = torch.tensor(tok.encode(new_text))
old_data = train_tokens()  # 事前学習と同じ漱石 + 加減算ドリル
# [end: cpt-data]


def evaluate(model) -> dict:
    """3 つの物差しで測る: 漱石の損失、掛け算と加減算の正答率。"""
    return {
        "soseki_loss": soseki_val_loss(model),
        "mul_acc": drill_accuracy(model, tok, n_problems=100, ops="*",
                                  max_value=99, device=device),
        "addsub_acc": drill_accuracy(model, tok, n_problems=100,
                                     device=device),
    }


model = load_base()
optimizer = make_optimizer(model, max_lr, weight_decay=0.1)
history = []
t0 = time.time()
# [snippet: cpt-loop]
n_new = round(batch_size * new_ratio)  # バッチのうち新領域から取る行数
for step in range(max_steps + 1):
    if step % 25 == 0:
        history.append({"step": step, **evaluate(model)})
        print(history[-1], f"{time.time() - t0:.0f} 秒", flush=True)
    if step == max_steps:
        break
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step, max_lr, warmup_steps=20,
                            max_steps=max_steps)
    x, y = get_batch(new_data, n_new, block_size, device)
    if n_new < batch_size:  # 残りの行を元のデータから取り、混ぜる
        x2, y2 = get_batch(old_data, batch_size - n_new, block_size, device)
        x, y = torch.cat([x, x2]), torch.cat([y, y2])
    model.train()
    _, loss = model(x, y)  # 損失は事前学習と同じ次トークン予測
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
# [end: cpt-loop]

with open(OUT_DIR / f"cpt_{name}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
save_checkpoint(model, CKPT_DIR / f"ch16_cpt_{name}.pt")
first, last = history[0], history[-1]
with open(OUT_DIR / f"cpt_{name}.txt", "w") as f:
    print(f"設定 {name}: 最大学習率 {max_lr:g}, 新領域の割合 {new_ratio:.0%}",
          file=f)
    for key, label in [("soseki_loss", "漱石の損失    "),
                       ("mul_acc", "掛け算の正答率"),
                       ("addsub_acc", "加減算の正答率")]:
        fmt = "{:.3f}" if key == "soseki_loss" else "{:.0%}"
        print(f"{label}: {fmt.format(first[key])} -> {fmt.format(last[key])}",
              file=f)
    for prompt in ["\n12+35=", "\n23*4="]:  # 学習後の答え方の例
        reply = complete(model, tok, prompt, device=device)
        print(f"{prompt.strip()}{reply}", file=f)
print(f"所要時間 {time.time() - t0:.0f} 秒")
