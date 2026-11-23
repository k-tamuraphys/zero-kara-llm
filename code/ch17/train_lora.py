"""base.pt に LoRA を差し込み、第 16 章と同じ SFT で計算アシスタントにする。

実行:  python ch17/train_lora.py [r] [targets] [lr] [tag]
  r: ランク（既定 8）  targets: all / qkv / attn / mlp（既定 all）
  lr: 最大学習率（既定 1e-3）  tag: 出力名（既定 r{r}_{targets}）
出力:  ch17/out/lora_{tag}.txt, lora_{tag}_loss.csv,
       checkpoints/ch17_lora_{tag}.pt（アダプタだけ）
"""

import sys
import time

import torch

from common import (CKPT_DIR, OUT_DIR, accuracy, load_base, sft_examples,
                    soseki_val_loss, train)
from lora import add_lora, count_params, lora_state_dict

TARGETS = {"all": ("attn.qkv", "attn.proj", "mlp.fc", "mlp.proj"),
           "qkv": ("attn.qkv",), "attn": ("attn.qkv", "attn.proj"),
           "mlp": ("mlp.fc", "mlp.proj")}
r = int(sys.argv[1]) if len(sys.argv) > 1 else 8
tname = sys.argv[2] if len(sys.argv) > 2 else "all"
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
tag = sys.argv[4] if len(sys.argv) > 4 else f"r{r}_{tname}"
torch.manual_seed(0)

# [snippet: run]
model = add_lora(load_base(), r=r, alpha=2 * r, targets=TARGETS[tname])
n_train, n_total = count_params(model)
print(f"r={r} 対象={tname} 学習率={lr}")
print(f"学習するパラメータ: {n_train:,} / {n_total:,} "
      f"({n_train / n_total:.2%})")
t0 = time.time()
log = []
train(model, sft_examples(), max_lr=lr, log=log)
print(f"学習時間: {time.time() - t0:.0f} 秒")
torch.save(lora_state_dict(model), CKPT_DIR / f"ch17_lora_{tag}.pt")
print(f"対話正答率（1,000 問）: {accuracy(model):.1%}")
print(f"漱石の検証損失: {soseki_val_loss(model):.3f}")
# [end: run]
size = (CKPT_DIR / f"ch17_lora_{tag}.pt").stat().st_size
print(f"アダプタのファイルサイズ: {size / 1024:.0f} KB")
with open(OUT_DIR / f"lora_{tag}_loss.csv", "w") as f:
    f.write("step,loss\n")
    f.writelines(f"{s},{l:.4f}\n" for s, l in log)
