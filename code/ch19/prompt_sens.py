"""プロンプト感度: 同じ計算を、言い回しだけ変えてたずねる。"""

import csv

from common import load, make_problems
from engine import Engine, reply_batch

# 学習データにある言い回し（演算ごとに 3 通り）と、学習データにない言い回し
PHRASINGS = [
    ("学習あり", "{a}{op}{b}は？", "{a}{op}{b}は？"),
    ("学習あり", "{a}足す{b}は？", "{a}引く{b}は？"),
    ("学習あり", "{a}と{b}を足すといくつですか？",
     "{a}から{b}を引くといくつですか？"),
    ("学習なし", "{a}{op}{b}はいくつ？", "{a}{op}{b}はいくつ？"),
    ("学習なし", "{a}{op}{b}=", "{a}{op}{b}="),
    ("学習なし", "{a} {op} {b} は？", "{a} {op} {b} は？"),
    ("学習なし", "{a}たす{b}は？", "{a}ひく{b}は？"),
    ("学習なし", "{a}{op}{b}を計算してください。",
     "{a}{op}{b}を計算してください。"),
]

model, tok = load("sft")
engine = Engine(model, n_slots=64)
problems = make_problems(1000, seed=777)  # 数と演算だけを使う

rows = []
for seen, plus, minus in PHRASINGS:
    questions = [(plus if p["op"] == "+" else minus).format(**p)
                 for p in problems]
    replies = []
    for i in range(0, len(questions), 64):
        replies += reply_batch(engine, tok, questions[i:i + 64])
    acc = sum(r == p["target"] for r, p in zip(replies, problems)) / 1000
    example = plus.format(a=12, b=35, op="+")
    rows.append((seen, example, acc, replies[0]))
    print(f"{seen}  {acc:6.1%}  {example}")

with open("ch19/out/prompt_sens.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seen", "example", "acc"])
    for seen, example, acc, _ in rows:
        w.writerow([seen, example, f"{100 * acc:.1f}"])
print("\n最初の問題への応答（参考）")
print(f"  問題: {problems[0]['a']}{problems[0]['op']}{problems[0]['b']}"
      f"  正解: {problems[0]['target']}")
for seen, example, acc, r0 in rows:
    print(f"  {r0!r}")
