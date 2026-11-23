"""正答率のぶれ: 標準誤差、信頼区間、ブートストラップ。"""

import csv
import math

import numpy as np

from common import evaluate


# [snippet: standard-error]
def standard_error(p: float, n: int) -> float:
    """正答率 p を n 問で測ったときの標準誤差（二項分布の式）。"""
    return math.sqrt(p * (1 - p) / n)


def confidence_interval(p: float, n: int, z: float = 1.96):
    """95% 信頼区間（正規分布による近似）。"""
    se = standard_error(p, n)
    return p - z * se, p + z * se
# [end: standard-error]


# [snippet: bootstrap]
def bootstrap_ci(correct: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """正誤の列 correct (n,) から、ブートストラップで 95% 信頼区間を求める。"""
    rng = np.random.default_rng(seed)
    n = len(correct)
    # n 問の中から、重複を許して n 問を選び直す。これを n_boot 回くり返す。
    picks = rng.integers(0, n, size=(n_boot, n))      # (n_boot, n)
    accs = correct[picks].mean(axis=1)                # 選び直すたびの正答率
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5), accs.std()
# [end: bootstrap]


if __name__ == "__main__":
    # (1) 二つの「公式記録」
    for n, seed in [(200, 12345), (1000, 777)]:
        rec = evaluate("sft", n, seed)
        p = sum(r["correct"] for r in rec) / n
        lo, hi = confidence_interval(p, n)
        print(f"{n:5d} 問（種 {seed:5d}）: 正答率 {p:.1%}  標準誤差 "
              f"{standard_error(p, n):.1%}  95% 信頼区間 [{lo:.1%}, {hi:.1%}]")

    # (2) 200 問の試験を、問題を変えて 50 回くり返す
    accs = []
    for seed in range(1, 51):
        rec = evaluate("sft", 200, seed)
        accs.append(sum(r["correct"] for r in rec) / 200)
    accs = np.array(accs)
    p_all = accs.mean()  # 1 万問ぶんの正答率
    print(f"\n200 問の試験 50 回: 最低 {accs.min():.1%}  最高 {accs.max():.1%}"
          f"  平均 {p_all:.1%}")
    print(f"  50 回の正答率の標準偏差（実測）: {accs.std(ddof=1):.2%}")
    print(f"  式 sqrt(p(1-p)/n) による予測   : "
          f"{standard_error(p_all, 200):.2%}")
    inside = np.abs(accs - p_all) <= 1.96 * standard_error(p_all, 200)
    print(f"  平均 ± 1.96 標準誤差 に入った回数: {inside.sum()} / 50")
    lo, hi = confidence_interval(p_all, 10_000)
    print(f"  1 万問まとめた 95% 信頼区間: [{lo:.1%}, {hi:.1%}]")

    # 図のための度数分布（幅 1 ポイント）と、正規分布による予測
    se = standard_error(p_all, 200)
    with open("ch19/out/hist200.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["acc", "count", "normal"])
        for c in range(74, 93):
            lo_c, hi_c = c - 0.5, c + 0.5
            count = int(((accs * 100 >= lo_c) & (accs * 100 < hi_c)).sum())
            z = (c / 100 - p_all) / se
            normal = (50 * 0.01 / se * math.exp(-z * z / 2)
                      / math.sqrt(2 * math.pi))
            w.writerow([c, count, f"{normal:.3f}"])

    # (3) ブートストラップ（1,000 問、種 777）
    correct = np.array([r["correct"] for r in evaluate("sft", 1000, 777)],
                       dtype=float)
    lo, hi, sd = bootstrap_ci(correct)
    flo, fhi = confidence_interval(correct.mean(), 1000)
    print(f"\n1,000 問のブートストラップ（1 万回）")
    print(f"  選び直した正答率の標準偏差: {sd:.2%}"
          f"（式: {standard_error(correct.mean(), 1000):.2%}）")
    print(f"  95% 信頼区間: [{lo:.1%}, {hi:.1%}]"
          f"（式: [{flo:.1%}, {fhi:.1%}]）")

    # (4) 問題数と誤差（正答率 85% のモデルを想定）
    print("\n問題数 n と誤差（p = 0.85 のとき）")
    with open("ch19/out/se_vs_n.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "se", "half"])
        for n in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000]:
            se = standard_error(0.85, n)
            w.writerow([n, f"{100 * se:.3f}", f"{196 * se:.3f}"])
            print(f"  n = {n:6d}: 標準誤差 {se:.2%}"
                  f"  95% 区間の幅 ±{1.96 * se:.1%}")
