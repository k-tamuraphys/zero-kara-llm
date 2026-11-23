"""言語モデルの学習ループと、その部品。"""

import csv
import math
import time
from pathlib import Path

import torch

from .model import GPT, GPTConfig


# [snippet: get-batch]
def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """トークン列 data からランダムな位置を選び、入力 x と正解 y を作る。

    y は x を 1 トークン後ろにずらしたもの（次トークン予測の正解）。
    """
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x.to(device), y.to(device)
# [end: get-batch]


# [snippet: estimate-loss]
@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, batch_size: int,
                  n_batches: int = 20, device: str = "cpu") -> float:
    """ランダムな n_batches 個のバッチで平均損失を見積もる。"""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, model.config.block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)
# [end: estimate-loss]


# [snippet: lr-at]
def lr_at(step: int, max_lr: float, warmup_steps: int, max_steps: int,
          min_lr_ratio: float = 0.1) -> float:
    """線形ウォームアップ + コサイン減衰の学習率スケジュール。"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
# [end: lr-at]


# [snippet: make-optimizer]
def make_optimizer(model: torch.nn.Module, lr: float,
                   weight_decay: float = 0.1) -> torch.optim.AdamW:
    """AdamW を作る。重み減衰は行列（2 次元以上）のパラメータだけにかける。"""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
# [end: make-optimizer]


# [snippet: train-lm]
def train_lm(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor,
             max_steps: int, batch_size: int = 32, max_lr: float = 1e-3,
             warmup_steps: int = 100, weight_decay: float = 0.1,
             grad_clip: float = 1.0, eval_interval: int = 250,
             device: str = "cpu", log_path=None) -> list[dict]:
    """次トークン予測で model を学習する。学習の記録を返す。"""
    model.to(device).train()
    optimizer = make_optimizer(model, max_lr, weight_decay)
    history = []
    t0 = time.time()
    for step in range(max_steps + 1):
        # 定期的に訓練損失と検証損失を測る
        if step % eval_interval == 0 or step == max_steps:
            record = {
                "step": step,
                "train_loss": estimate_loss(model, train_data, batch_size,
                                            device=device),
                "val_loss": estimate_loss(model, val_data, batch_size,
                                          device=device),
                "time": time.time() - t0,
            }
            history.append(record)
            print(f"step {step:5d} | train {record['train_loss']:.3f} | "
                  f"val {record['val_loss']:.3f} | {record['time']:.0f} 秒")
        if step == max_steps:
            break
        lr = lr_at(step, max_lr, warmup_steps, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, batch_size, model.config.block_size,
                         device)
        _, loss = model(x, y)  # 順伝播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # 逆伝播
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()  # パラメータ更新
    if log_path is not None:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return history
# [end: train-lm]


# [snippet: checkpoint]
def save_checkpoint(model: GPT, path, **extra) -> None:
    """モデルの設定と重みを 1 つのファイルに保存する。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(model.config),
                "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device: str = "cpu") -> GPT:
    """save_checkpoint で保存したモデルを読み込む。"""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)
# [end: checkpoint]
