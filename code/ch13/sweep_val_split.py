"""base.pt の検証損失を、小説の部分とドリルの部分に分けて測る。

実行: python ch13/sweep_val_split.py  （2 スレッドで約 1 分）
出力: ch13/out/sweep_val_split.txt
"""

import torch

from ch13.common import load_bin
from llmbook.corpus import DATA_DIR
from llmbook.train import load_checkpoint

torch.set_num_threads(2)
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt").eval()
T = model.config.block_size  # 128
EOT = 4088  # <|endoftext|> の ID


@torch.no_grad()
def mean_loss(data: torch.Tensor) -> float:
    """重ならない長さ T の窓をすべて使い、トークンあたりの平均損失を返す。"""
    n_win = (len(data) - 1) // T
    total = 0.0
    for i in range(0, n_win, 64):  # 64 窓ずつまとめて計算する
        s = torch.arange(i, min(i + 64, n_win)) * T
        x = torch.stack([data[j:j + T] for j in s])
        y = torch.stack([data[j + 1:j + T + 1] for j in s])
        _, loss = model(x, y)
        total += loss.item() * len(s)
    return total / n_win


data = load_bin("val")
cut = int((data == EOT).nonzero()[0])  # 小説とドリルの境目
novel, drill = data[:cut], data[cut + 1:-1]
print(f"検証データ: 小説 {len(novel):,} トークン / ドリル {len(drill):,}"
      f" トークン（{len(drill) / len(data):.0%}）")
print(f"損失: 小説 {mean_loss(novel):.3f} / ドリル {mean_loss(drill):.3f}"
      f" / 全体 {mean_loss(data):.3f}")
