"""モデルの大きさを 3 通りに変えて、同じ条件で短く学習し損失を比べる。

実行: python ch13/scaling_experiment.py small   （small / medium / large）
      1 スレッドで small 約 1 分、medium 約 2 分、large 約 4 分
出力: ch13/out/scaling_{name}.txt, ch13/out/scaling_{name}.csv
"""

import sys

import torch

from ch13.common import load_bin
from llmbook.model import GPT, GPTConfig
from llmbook.train import estimate_loss, train_lm

torch.set_num_threads(1)
train_data, val_data = load_bin("train"), load_bin("val")

# [snippet: sizes]
SIZES = {  # 名前: (層数 L, 埋め込み次元 d)
    "small": (1, 32),
    "medium": (2, 64),
    "large": (4, 128),
}
name = sys.argv[1]
n_layer, n_embd = SIZES[name]
torch.manual_seed(0)  # 3 つとも同じ乱数で始める
model = GPT(GPTConfig(block_size=64, n_layer=n_layer, n_head=4,
                      n_embd=n_embd))
n_emb = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
train_lm(model, train_data, val_data, max_steps=1000, batch_size=16,
         max_lr=2e-3, warmup_steps=100, eval_interval=250,
         log_path=f"ch13/out/scaling_{name}.csv")
torch.manual_seed(123)  # 最後の評価は 3 つとも同じ 100 バッチで行う
val = estimate_loss(model, val_data, 16, n_batches=100)
print(f"{name}: 全体 {model.num_params():,} / 埋め込み以外 "
      f"{model.num_params() - n_emb:,} / 検証損失 {val:.3f}")
# [end: sizes]
