"""二つのモデルの比較: 対応のない比較と、対応のある比較。"""

import math
import sys

import numpy as np

from common import evaluate
from stats import standard_error


# [snippet: unpaired]
def unpaired(p_a: float, n_a: int, p_b: float, n_b: int):
    """別々の問題集で測った二つの正答率の差と、その標準誤差。"""
    se = math.sqrt(standard_error(p_a, n_a) ** 2
                   + standard_error(p_b, n_b) ** 2)
    return p_b - p_a, se
# [end: unpaired]


# [snippet: paired]
def paired(a: np.ndarray, b: np.ndarray):
    """同じ n 問に対する正誤の列 a, b (n,) から、差とその標準誤差を求める。"""
    n = len(a)
    only_a = int((a & ~b).sum())  # A だけが正解した問題の数
    only_b = int((~a & b).sum())  # B だけが正解した問題の数
    diff = (only_b - only_a) / n  # 正答率の差（両方正解・両方誤りは打ち消す）
    se = math.sqrt(only_a + only_b - (only_b - only_a) ** 2 / n) / n
    return only_a, only_b, diff, se
# [end: paired]


# [snippet: sign-test]
def sign_test(only_a: int, only_b: int) -> float:
    """「実力が同じなら、食い違いはコイン投げ」と考えたときの p 値。"""
    m, k = only_a + only_b, max(only_a, only_b)  # 投げた回数と、多いほうの数
    # m 回のうち i 回が表になる確率は comb(m, i) / 2**m。これを i >= k で足す
    tail = sum(math.comb(m, i) for i in range(k, m + 1)) / 2 ** m
    return min(1.0, 2 * tail)  # 表が多い場合と裏が多い場合で、2 倍
# [end: sign-test]


def report(name_a: str, name_b: str, sets: list[tuple[int, int]]) -> None:
    """問題集 (問題数, 種) のリストを全部つないで、A と B を比べる。"""
    a = np.array([r["correct"] for n, seed in sets
                  for r in evaluate(name_a, n, seed)])
    b = np.array([r["correct"] for n, seed in sets
                  for r in evaluate(name_b, n, seed)])
    n = len(a)
    print(f"A = {name_a}.pt: {a.mean():.2%}   B = {name_b}.pt: {b.mean():.2%}"
          f"   （{n} 問）")
    d, se = unpaired(a.mean(), n, b.mean(), n)
    print(f"  対応なし: 差 {d:+.2%}  標準誤差 {se:.2%}  差 / 標準誤差 = "
          f"{d / se:+.2f}")
    only_a, only_b, d, se = paired(a, b)
    both = int((a & b).sum())
    print(f"  両方正解 {both}  A だけ正解 {only_a}  B だけ正解 {only_b}"
          f"  両方誤り {n - both - only_a - only_b}")
    print(f"  対応あり: 差 {d:+.2%}  標準誤差 {se:.2%}  差 / 標準誤差 = "
          f"{d / se:+.2f}")
    print(f"  符号検定の p 値: {sign_test(only_a, only_b):.4f}")
    # 対応のあるブートストラップ: 問題を選び直し、同じ問題で両方を採点する
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        picks = rng.integers(0, n, size=n)
        diffs.append(b[picks].mean() - a[picks].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  差の 95% 信頼区間（ブートストラップ）: [{lo:+.2%}, {hi:+.2%}]")


if __name__ == "__main__":
    name_b = sys.argv[1] if len(sys.argv) > 1 else "ch16_sft"
    print("(1) 固定の 1,000 問（種 777）")
    report("sft", name_b, [(1000, 777)])
    print("\n(2) 新しい 200 問 × 50 組 = 1 万問（種 1〜50）")
    report("sft", name_b, [(200, seed) for seed in range(1, 51)])
