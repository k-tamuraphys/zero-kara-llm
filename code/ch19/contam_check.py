"""汚染の実験（contamination.py）の結果を、二つの点で確かめ直す。

(1) 「ドリルで見た計算」の双子は、なぜ正答率が高いのか。
    小さいほうの数の桁数を数える。
(2) 試験問題を混ぜて追加学習したモデル（ch19_leak.pt）は、新しい問題で
    本当に悪くなったのか。対応のある比較で確かめる。
"""

import re
from collections import Counter

import numpy as np

from common import evaluate, make_problems
from compare import paired, sign_test
from llmbook.arith import drill_text


def key(p: dict) -> tuple:
    return (p["a"], p["op"], p["b"])


def small_digits(p: dict) -> int:
    """小さいほうの数の桁数。"""
    return len(str(min(p["a"], p["b"])))


sft_train = make_problems(3000, seed=0)
train_keys = {key(p) for p in sft_train}
drill_keys = set()
for line in drill_text(120_000, seed=0).splitlines():
    a, op, b = re.match(r"(\d+)([+-])(\d+)=", line).groups()
    drill_keys.add((int(a), op, int(b)))

print("(1) 3 桁の問題 1,000 問の、小さいほうの数の桁数")
sft_seen = list({p["question"]: p for p in sft_train
                 if p["digits"] == 3}.values())[:1000]
pool = [p for p in make_problems(20000, seed=5)
        if p["digits"] == 3 and key(p) in drill_keys
        and key(p) not in train_keys]
drill_seen = list({key(p): p for p in pool}.values())[:1000]
for name, ps in [("SFT で見た質問  ", sft_seen),
                 ("ドリルで見た計算", drill_seen)]:
    c = Counter(small_digits(p) for p in ps)
    print(f"  {name}: 1 桁 {c[1]:3d} 問  2 桁 {c[2]:3d} 問  3 桁 {c[3]:3d} 問")

print("\n(2) 新しい 1,000 問（種 778）での、追加学習の前後の比較")
a = np.array([r["correct"] for r in evaluate("sft", 1000, 778)])
b = np.array([r["correct"] for r in evaluate("ch19_leak", 1000, 778)])
only_a, only_b, diff, se = paired(a, b)
print(f"  前 {a.mean():.1%}  後 {b.mean():.1%}"
      f"  前だけ正解 {only_a}  後だけ正解 {only_b}")
print(f"  対応あり: 差 {diff:+.2%}  標準誤差 {se:.2%}"
      f"  差 / 標準誤差 = {diff / se:+.2f}"
      f"  符号検定の p 値 {sign_test(only_a, only_b):.3f}")
