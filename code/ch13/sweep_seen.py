"""base.pt の計算ドリルの正答率を、訓練データに同じ行があるかで分ける。

実行: python ch13/sweep_seen.py  （2 スレッドで約 2 分）
出力: ch13/out/sweep_seen.txt, ch13/out/sweep_seen.csv
"""

import random

import torch

from llmbook.arith import complete, drill_text, make_problem
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.set_num_threads(2)
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt").eval()
# 訓練用ドリル（prepare_data.py と同じ種）の行の集合
seen = set(drill_text(120_000, seed=0).splitlines())


# arith_analysis.py の kind_of と同じもの（import すると評価が走るので写す）
def kind_of(question: str) -> tuple[str, str]:
    """問題を（演算と最大桁数, 繰り上がり・繰り下がりの有無）に分類する。"""
    op = "+" if "+" in question else "-"
    a, b = (int(s) for s in question[:-1].split(op))
    digits = max(len(str(a)), len(str(b)))
    da, db = str(a)[::-1], str(b)[::-1]  # 1 の位から見る
    carry = False
    for k in range(digits):
        x = int(da[k]) if k < len(da) else 0
        y = int(db[k]) if k < len(db) else 0
        carry |= (x + y >= 10) if op == "+" else (x < y)
    return f"{op}{digits}桁", "あり" if carry else "なし"


def evaluate(n: int, seed: int) -> list[tuple[str, bool, bool]]:
    """(問題, 既出か, 正解か) の一覧を返す。問題の作り方は本文と同じ。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        question, answer = make_problem(rng)
        ok = complete(model, tok, "\n" + question) == answer
        out.append((question, question + answer in seen, ok))
    return out


def report(name: str, results) -> None:
    for label, flag in [("既出", True), ("初出", False)]:
        sub = [ok for _, s, ok in results if s == flag]
        print(f"{name} {label}: {sum(sub)}/{len(sub)}"
              f" = {sum(sub) / len(sub):.1%}")


r200 = evaluate(200, 12345)  # drill_accuracy と同じ 200 問
print(f"200 問: {sum(ok for *_, ok in r200)}/200")
report("200 問", r200)
r1000 = evaluate(1000, 2024)  # arith_analysis.py と同じ 1,000 問
print(f"1,000 問: {sum(ok for *_, ok in r1000)}/1000")
report("1,000 問", r1000)

# 種類（演算と大きいほうの桁数）× 既出・初出 の正答率
with open("ch13/out/sweep_seen.csv", "w") as f:
    f.write("kind,seen_n,seen_acc,new_n,new_acc\n")
    for kind in ["+1桁", "+2桁", "+3桁", "-1桁", "-2桁", "-3桁"]:
        cells = []
        for flag in [True, False]:
            sub = [ok for q, s, ok in r1000
                   if kind_of(q)[0] == kind and s == flag]
            acc = 100 * sum(sub) / len(sub) if sub else float("nan")
            cells += [str(len(sub)), f"{acc:.1f}"]
        f.write(f"{kind},{','.join(cells)}\n")
        print(kind, "既出", cells[0], cells[1], "/ 初出", cells[2], cells[3])

# 二つの数の桁数が同じか違うか（1,000 問）
for label, same in [("桁数が同じ", True), ("桁数が違う", False)]:
    sub = []
    for q, _, ok in r1000:
        a, b = q[:-1].replace("-", "+").split("+")
        if (len(a) == len(b)) == same:
            sub.append(ok)
    print(f"{label}: {sum(sub)}/{len(sub)} = {sum(sub) / len(sub):.1%}")

# 大きいほうが 3 桁の問題を、二つの数の桁数の組で分ける（1,000 問）
for pair in ["1桁と3桁", "2桁と3桁", "3桁と3桁"]:
    sub = []
    for q, _, ok in r1000:
        a, b = q[:-1].replace("-", "+").split("+")
        small, large = sorted((len(a), len(b)))
        if f"{small}桁と{large}桁" == pair:
            sub.append(ok)
    print(f"{pair}: {sum(sub)}/{len(sub)} = {sum(sub) / len(sub):.1%}")
