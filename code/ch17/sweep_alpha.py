"""α を 16 に固定してランク r を変える（本文の α=2r との比較用）。

本文の実験は α=2r なので、係数 α/r はいつも 2 だった。このため
「r を上げると忘却が増える」には、方向の数が増えた効果と、差分が
大きくなった効果が混ざる。ここでは α=16 に固定し（係数 = 16/r）、
r だけを変えたときに、正答率と漱石の損失がどう変わるかを測る。
r=8 は α=2r と同じ設定（係数 2）なので、本文の結果を使う。
ほかの条件（全線形層、学習率 1e-3、600 ステップ、種 0）は本文と同じ。

実行:  python ch17/sweep_alpha.py r        （1 回に 1 つの r。数分）
       python ch17/sweep_alpha.py collect  （結果を CSV にまとめる）
出力:  ch17/out/sweep_alpha_r{r}.txt（標準出力を保存）,
       ch17/out/sweep_alpha_r{r}_loss.csv, ch17/out/sweep_alpha.csv
"""

import re
import sys
import time

import torch

from common import OUT_DIR, accuracy, load_base, sft_examples, \
    soseki_val_loss, train
from lora import add_lora, count_params

ALPHA = 16.0  # 固定する α（r=8 のとき α=2r と一致する）
RANKS = [1, 2, 4, 8, 16, 64]

if sys.argv[1] == "collect":
    # r=8 は本文の実行結果（α=2r=16 と同じ設定）を使う
    rows = []
    for r in RANKS:
        f = OUT_DIR / (f"lora_r{r}_all.txt" if r == 8
                       else f"sweep_alpha_r{r}.txt")
        s = f.read_text()
        acc = float(re.search(r"正答率（1,000 問）: ([\d.]+)%", s).group(1))
        loss = float(re.search(r"検証損失: ([\d.]+)", s).group(1))
        rows.append(f"{r},{ALPHA / r:g},{acc},{loss}")
    text = "r,scale,acc,soseki\n" + "\n".join(rows) + "\n"
    (OUT_DIR / "sweep_alpha.csv").write_text(text)
    print(text, end="")
    sys.exit()

r = int(sys.argv[1])
torch.manual_seed(0)  # train_lora.py と同じ種
model = add_lora(load_base(), r=r, alpha=ALPHA)  # 全線形層
n_train, _ = count_params(model)
print(f"r={r} alpha={ALPHA:g} 係数 alpha/r={ALPHA / r:g} 学習率=0.001")
print(f"学習するパラメータ: {n_train:,}")
t0 = time.time()
log = []
train(model, sft_examples(), max_lr=1e-3, log=log)  # 600 ステップ
print(f"学習時間: {time.time() - t0:.0f} 秒")
print(f"対話正答率（1,000 問）: {accuracy(model):.1%}")
print(f"漱石の検証損失: {soseki_val_loss(model):.3f}")
with open(OUT_DIR / f"sweep_alpha_r{r}_loss.csv", "w") as f:
    f.write("step,loss\n")
    f.writelines(f"{s},{l:.4f}\n" for s, l in log)
