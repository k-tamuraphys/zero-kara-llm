"""第 17 章の実行結果を集めて、「学ぶ量と忘れる量」の図用の CSV を作る。

学習はしない（out/*.txt を読むだけ。数秒）。図は本文の pgfplots が描く。
実行:  python ch17/fig_tradeoff.py
出力:  ch17/out/tradeoff.csv（group,label,tag,n_train,acc,soseki）
       ch17/out/tradeoff_{group}.csv（同じものを group ごとに分けたもの）
tag: 図で点の横に書く文字（重なる点は空）
group: rank（α=2r で r を変える）, alpha16（α=16 固定で r を変える）,
       lr（r=8 で学習率を変える）, target（r=8 で付ける層を変える）,
       full（全パラメータ SFT）
"""

import re

from common import OUT_DIR

RUNS = [  # (group, label, ファイル名)
    *[("rank", f"{r}", f"lora_r{r}_all.txt")
      for r in [1, 2, 4, 8, 16, 32, 64]],
    *[("alpha16", f"{r}", "lora_r8_all.txt" if r == 8
       else f"sweep_alpha_r{r}.txt") for r in [1, 2, 4, 8, 16, 64]],
    ("lr", "3e-4", "lora_r8_all_lr3e-4.txt"),
    ("lr", "1e-3", "lora_r8_all.txt"),
    ("lr", "3e-3", "lora_r8_all_lr3e-3.txt"),
    ("target", "qkv", "lora_r8_qkv.txt"),
    ("target", "attn", "lora_r8_attn.txt"),
    ("target", "mlp", "lora_r8_mlp.txt"),
    ("full", "SFT", "full_sft_ref.txt"),
]


def read(name):
    """出力ファイルから (学習するパラメータ数, 正答率, 漱石の損失)。"""
    f = OUT_DIR / name
    if not f.exists():
        return None
    s = f.read_text()
    m = re.search(r"正答率（1,000 問）: ([\d.]+)%", s)
    if m is None:  # 途中で止まった実行は使わない
        return None
    n = re.search(r"パラメータ: ([\d,]+)", s)
    n = int(n.group(1).replace(",", "")) if n else 0
    loss = float(re.search(r"検証損失: ([\d.]+)", s).group(1))
    return n, float(m.group(1)), loss


rows, groups = [], {}
for group, label, name in RUNS:
    v = read(name)
    if v is None:
        print(f"（なし）{name}")
        continue
    # 図で重なる点（r=8 と同じ位置の点）には、横に添える文字を付けない
    crowd = (group == "alpha16" and label in ("8", "16", "64")) \
        or (group == "lr" and label == "1e-3")
    row = f"{label},{'' if crowd else label},{v[0]},{v[1]},{v[2]}"
    rows.append(f"{group},{row}")
    groups.setdefault(group, []).append(row)
(OUT_DIR / "tradeoff.csv").write_text(
    "group,label,tag,n_train,acc,soseki\n" + "\n".join(rows) + "\n")
for group, g in groups.items():  # pgfplots で系列ごとに読む
    (OUT_DIR / f"tradeoff_{group}.csv").write_text(
        "label,tag,n_train,acc,soseki\n" + "\n".join(g) + "\n")
print("\n".join(rows))
