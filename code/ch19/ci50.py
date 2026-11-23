"""信頼区間を定義どおりに確かめる: 50 個の区間のうち、何個が真の値を含むか。

stats.py と同じ 200 問 × 50 組（種 1〜50）の結果を使う（out/cache から読む）。
"""

import csv
import math

from common import evaluate
from stats import confidence_interval, standard_error

accs, all_records = [], []
for seed in range(1, 51):
    rec = evaluate("sft", 200, seed)
    accs.append(sum(r["correct"] for r in rec) / 200)
    all_records += rec
p_all = sum(accs) / 50  # 1 万問ぶんの正答率。これを「真の値」の代わりにする

# [snippet: ci50]
n_hit, rows = 0, []
for seed, p in enumerate(accs, start=1):
    lo, hi = confidence_interval(p, 200)  # その 200 問だけから作った区間
    hit = lo <= p_all <= hi               # 区間は「真の値」を含んだか
    n_hit += hit
    rows.append((seed, p, lo, hi, hit))
# [end: ci50]

print(f"1 万問の正答率（真の値の代わり）: {p_all:.2%}")
print(f"200 問の 95% 信頼区間 50 個のうち、これを含んだもの: {n_hit} 個")
for seed, p, lo, hi, hit in rows:
    if not hit:
        print(f"  外れた区間: 種 {seed:2d}  正答率 {p:.1%}"
              f"  区間 [{lo:.1%}, {hi:.1%}]")

with open("ch19/out/ci50.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "acc", "half", "hit_acc", "miss_acc"])
    for seed, p, lo, hi, hit in rows:
        acc, half = f"{100 * p:.1f}", f"{100 * (hi - p):.2f}"
        w.writerow([seed, acc, half, acc if hit else "nan",
                    "nan" if hit else acc])

# 二つの「公式記録」は、真の値からどれだけ離れているか
print("\n真の値を 1 万問の値としたときの、二つの記録の外れ方")
for n, seed in [(200, 12345), (1000, 777)]:
    rec = evaluate("sft", n, seed)
    k = sum(r["correct"] for r in rec)
    se = standard_error(p_all, n)  # 真の値から計算した標準誤差
    z = (k / n - p_all) / se
    # 正規分布で、これ以上（どちら向きでも）外れる確率
    two_sided = math.erfc(abs(z) / math.sqrt(2))
    # 二項分布で、正解が k 問以下になる確率（近似なしの計算）
    lower = sum(math.comb(n, i) * p_all ** i * (1 - p_all) ** (n - i)
                for i in range(k + 1))
    print(f"  {n:4d} 問で {k / n:.1%}: 標準誤差 {se:.2%}"
          f"  ずれはその {z:+.2f} 倍"
          f"  これ以上外れる確率 {two_sided:.4f}"
          f"（約 {1 / two_sided:.0f} 回に 1 回）"
          f"  {k} 問以下になる確率 {lower:.4f}")

# 種 777 の 1,000 問は、どこが難しかったのか。3 桁の問題の割合と正答率を比べる
print("\n3 桁の問題の割合と、その正答率")
for name, rec in [("1 万問（種 1〜50）", all_records),
                  ("1,000 問（種 777）", evaluate("sft", 1000, 777))]:
    three = [r["correct"] for r in rec if r["digits"] == 3]
    print(f"  {name}: 3 桁の問題は {len(three) / len(rec):.1%}"
          f"  その正答率 {sum(three) / len(three):.1%}")
