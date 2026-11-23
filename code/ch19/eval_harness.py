"""評価ハーネス: 問題集 → 応答の生成 → 採点 → 集計 → 誤りの分析。"""

import csv
import re
from collections import Counter

from common import load, make_problems
from llmbook.chat import chat_reply


# [snippet: run-eval]
def run_eval(reply_fn, problems: list[dict]) -> list[dict]:
    """各問題に応答させて採点する。reply_fn は「質問 → 応答」の関数。"""
    records = []
    for p in problems:
        reply = reply_fn(p["question"])
        # 採点: 模範応答と 1 文字も違わなければ正解（完全一致）
        records.append({**p, "reply": reply, "correct": reply == p["target"]})
    return records
# [end: run-eval]


def breakdown(records: list[dict], key) -> dict:
    """key(記録) で分類し、分類ごとの (正解数, 問題数) を返す。"""
    table = {}
    for r in records:
        ok, n = table.get(key(r), (0, 0))
        table[key(r)] = (ok + r["correct"], n + 1)
    return dict(sorted(table.items()))


# [snippet: error-kind]
def error_kind(r: dict) -> str:
    """誤答を種類に分ける。"""
    m = re.fullmatch(r"(\d+)です。", r["reply"])
    if m is None:
        return "形式が違う"        # 「◯◯です。」の形になっていない
    got, want = m.group(1), r["target"][:-3]
    if len(got) != len(want):
        return "桁数が違う"
    n_diff = sum(g != w for g, w in zip(got, want))
    return "1 文字違い" if n_diff == 1 else "2 文字以上違う"
# [end: error-kind]


if __name__ == "__main__":
    model, tok = load("sft")
    problems = make_problems(1000, seed=777)
    records = run_eval(lambda q: chat_reply(model, tok, q), problems)
    n_ok = sum(r["correct"] for r in records)
    print(f"全体: {n_ok} / {len(records)} = {n_ok / len(records):.1%}")

    def line(title: str, key, names: dict) -> None:
        """内訳を 1 行で表示する: 「名前 正答率（問題数）」を並べる。"""
        cells = [f"{names[k]} {ok / n:.1%}（{n} 問）"
                 for k, (ok, n) in breakdown(records, key).items()]
        print(f"{title}:  " + "  ".join(cells))

    line("演算別", lambda r: r["op"], {"+": "+", "-": "-"})
    line("桁数別", lambda r: r["digits"], {1: "1 桁", 2: "2 桁", 3: "3 桁"})
    line("繰り上がり・繰り下がり", lambda r: r["carry"],
         {False: "なし", True: "あり"})

    # 図のための CSV（演算と桁数の組ごとに、繰り上がりなし・ありの正答率）
    table = breakdown(records, lambda r: (r["op"], r["digits"], r["carry"]))
    with open("ch19/out/by_category.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "op", "digits", "none", "carry"])
        for x, (op, d) in enumerate((op, d) for op in "+-" for d in (1, 2, 3)):
            accs = [f"{100 * table[op, d, c][0] / table[op, d, c][1]:.1f}"
                    if (op, d, c) in table else "nan" for c in (False, True)]
            w.writerow([x, op, d, *accs])

    errors = [r for r in records if not r["correct"]]
    kinds = Counter(error_kind(r) for r in errors).most_common()
    print(f"誤答 {len(errors)} 件の内訳:  "
          + "  ".join(f"{kind} {n} 件" for kind, n in kinds))
    print("誤答の例（質問 → 応答 [正解]）")
    for r in errors[:3]:
        print(f"  {r['question']} → {r['reply']} [{r['target']}]")
