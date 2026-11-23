"""第 5 章: deep_train.py を乱数の種 0〜4 で実行した結果を集計する。

「学習できた」は、訓練データの正解率が 99% 以上になったことを指す。
    cd code && python ch05/deep_summary.py > ch05/out/deep_summary.txt
"""
import csv

SEEDS = range(5)
NAMES = [
    ("default", "PyTorch の標準の初期値"),
    ("he", "He の初期化"),
    ("he_ln", "He＋レイヤー正規化"),
    ("he_res", "He＋残差接続"),
    ("he_res_ln", "He＋レイヤー正規化＋残差接続"),
]


def pad(s, width):
    return s + " " * (width - sum(2 if ord(c) > 255 else 1 for c in s))


ok, diverged = {}, {}
for seed in SEEDS:
    with open(f"ch05/out/deep_train_seed{seed}.csv") as f:
        for row in csv.DictReader(f):
            k = (int(row["depth"]), row["config"])
            acc = float(row["train_acc"])            # 発散したときは nan
            ok[k] = ok.get(k, 0) + (acc >= 0.99)
            diverged[k] = diverged.get(k, 0) + (acc != acc)

print(f"乱数の種 {len(SEEDS)} 通りのうち、学習できた回数"
      "（括弧内は発散した回数）")
print(pad("設定", 30) + "   10 枚    20 枚    50 枚")
for key, name in NAMES:
    cells = [f"{ok[(d, key)]}（{diverged[(d, key)]}）" for d in (10, 20, 50)]
    print(pad(name, 30) + "   " + "   ".join(cells))
