"""乱数の種だけを変えて、r=8 の LoRA の結果がどれだけぶれるかを測る。

本文の実験は種 0 の 1 回だけだった。ここでは、A の初期値（torch の種）と
学習データの並び（train の seed）を同じ種 s にそろえて変え、ほかの条件
（全線形層、r=8、α=16、学習率 1e-3、600 ステップ）は本文と同じにする。
評価の 1,000 問（種 777）は変えない。種 0 は本文の結果を使う。

実行:  python ch17/sweep_seed.py s        （1 回に 1 つの種。数分）
       python ch17/sweep_seed.py collect  （結果を CSV にまとめる）
出力:  ch17/out/sweep_seed_s{s}.txt（標準出力を保存）,
       ch17/out/sweep_seed.csv
"""

import re
import sys
import time

import torch

from common import OUT_DIR, accuracy, load_base, sft_examples, \
    soseki_val_loss, train
from lora import add_lora

SEEDS = [0, 1, 2]

if sys.argv[1] == "collect":
    rows = []
    for s in SEEDS:
        f = OUT_DIR / ("lora_r8_all.txt" if s == 0
                       else f"sweep_seed_s{s}.txt")
        t = f.read_text()
        acc = float(re.search(r"正答率（1,000 問）: ([\d.]+)%", t).group(1))
        loss = float(re.search(r"検証損失: ([\d.]+)", t).group(1))
        rows.append(f"{s},{acc},{loss}")
    text = "seed,acc,soseki\n" + "\n".join(rows) + "\n"
    (OUT_DIR / "sweep_seed.csv").write_text(text)
    print(text, end="")
    sys.exit()

s = int(sys.argv[1])
torch.manual_seed(s)  # A の初期値が変わる
model = add_lora(load_base(), r=8, alpha=16)  # 全線形層
print(f"種={s} r=8 alpha=16 学習率=0.001")
t0 = time.time()
train(model, sft_examples(), max_lr=1e-3, seed=s)  # バッチの並びも変わる
print(f"学習時間: {time.time() - t0:.0f} 秒")
print(f"対話正答率（1,000 問）: {accuracy(model):.1%}")
print(f"漱石の検証損失: {soseki_val_loss(model):.3f}")
