"""継続事前学習で、新しいデータ（掛け算ドリル）の割合を変えて比べる。

continual_pretrain.py と同じ学習（最大学習率 1e-3、300 ステップ、
バッチ 32 行 x 128 トークン）を、新データの割合だけ変えて行う。
途中の評価はせず、学習の後に一度だけ 3 つの物差しで測る。
割合 1.0 と 0.5 は、本文の high_lr と mix と同じ設定になる。
実行:  python ch16/sweep_cpt_mix.py 0.75   （1.0, 0.75, 0.5, 0.25 を 1 本ずつ）
出力:  ch16/out/sweep_cpt_mix.csv に 1 行を足す（同じ割合の行は置き換える）
       1 本あたり CPU で約 5 分。
"""

import csv
import sys
import time

import torch

from common import OUT_DIR, device, load_base, soseki_val_loss, tok
from common import train_tokens
from llmbook.arith import drill_accuracy, drill_text
from llmbook.train import get_batch, lr_at, make_optimizer

new_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
max_lr, max_steps, batch_size, block_size = 1e-3, 300, 32, 128

torch.manual_seed(0)  # continual_pretrain.py と同じ順に乱数を使う
new_text = drill_text(60_000, seed=7, max_value=99, ops="*")
new_data = torch.tensor(tok.encode(new_text))
old_data = train_tokens()

model = load_base()
optimizer = make_optimizer(model, max_lr, weight_decay=0.1)
t0 = time.time()
n_new = round(batch_size * new_ratio)  # バッチのうち新領域から取る行数
for step in range(max_steps):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step, max_lr, warmup_steps=20,
                            max_steps=max_steps)
    x, y = get_batch(new_data, n_new, block_size, device)
    if n_new < batch_size:  # 残りの行を元のデータから取り、混ぜる
        x2, y2 = get_batch(old_data, batch_size - n_new, block_size, device)
        x, y = torch.cat([x, x2]), torch.cat([y, y2])
    model.train()
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

row = {"new_ratio": new_ratio, "lam": 1 - new_ratio,  # lam は式の λ
       "mul_acc": drill_accuracy(model, tok, n_problems=100, ops="*",
                                 max_value=99, device=device),
       "addsub_acc": drill_accuracy(model, tok, n_problems=100,
                                    device=device),
       "soseki_loss": round(soseki_val_loss(model), 4)}
print(row, f"{time.time() - t0:.0f} 秒")

path = OUT_DIR / "sweep_cpt_mix.csv"
rows = list(csv.DictReader(open(path))) if path.exists() else []
rows = [r for r in rows if float(r["new_ratio"]) != new_ratio] + [row]
rows.sort(key=lambda r: float(r["new_ratio"]))
with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerows(rows)
