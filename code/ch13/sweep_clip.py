"""勾配クリップの有無と最大学習率を変えて、lr_sweep.py と同じ学習をする。

実行: python ch13/sweep_clip.py  （2 スレッドで約 3 分）
出力: ch13/out/sweep_clip.txt, ch13/out/clip_{学習率}_{on|off}.csv
      （列は step, loss, loss_ema, grad_norm）
"""

import csv

import torch

from ch13.common import load_bin
from llmbook.model import GPT, GPTConfig
from llmbook.train import estimate_loss, get_batch, lr_at, make_optimizer

torch.set_num_threads(2)
train_data, val_data = load_bin("train"), load_bin("val")
MAX_STEPS = 300

for max_lr in [1e-3, 1e-2, 1e-1]:
    for clip in [True, False]:
        torch.manual_seed(0)  # lr_sweep.py と同じ初期値・同じバッチの順
        model = GPT(GPTConfig(block_size=64, n_layer=2, n_head=4,
                              n_embd=64))
        optimizer = make_optimizer(model, max_lr)
        rows, ema = [], None
        for step in range(MAX_STEPS):
            lr = lr_at(step, max_lr, warmup_steps=30, max_steps=MAX_STEPS)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch(train_data, 16, 64)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # クリップしないときも、ノルムだけは測る（上限を無限大にする）
            limit = 1.0 if clip else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       limit)
            optimizer.step()
            v = loss.item()
            ema = v if ema is None else 0.9 * ema + 0.1 * v
            rows.append({"step": step, "loss": round(v, 4),
                         "loss_ema": round(ema, 4),
                         "grad_norm": round(grad_norm.item(), 4)})
        val = estimate_loss(model, val_data, 16)
        tag = f"{max_lr:g}_{'on' if clip else 'off'}"
        with open(f"ch13/out/clip_{tag}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        last = sum(r["loss"] for r in rows[-20:]) / 20
        peak = max(r["grad_norm"] for r in rows[10:])
        print(f"最大学習率 {max_lr:g} クリップ{'あり' if clip else 'なし'}:"
              f" 最後の 20 ステップの訓練損失 {last:.3f} / 検証損失 "
              f"{val:.3f} / 勾配ノルムの最大 {peak:.2f}")
