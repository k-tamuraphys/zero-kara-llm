"""計算ドリル 12 万行の重複を、二つの数の桁数の組ごとに数える。

実行: python ch13/sweep_dedup.py  （数秒）
出力: ch13/out/sweep_dedup.txt
"""

from collections import Counter

from llmbook.arith import drill_text


def digit_pair(line: str) -> str:
    """"12+5=17" → "1桁と2桁" のように、桁数の組（小さい順）を返す。"""
    op = "+" if "+" in line else "-"
    a, b = line.split("=")[0].split(op)
    small, large = sorted((len(a), len(b)))
    return f"{small}桁と{large}桁"


lines = drill_text(120_000, seed=0).splitlines()  # data_pipeline.py と同じ
count = Counter(lines)
rows = {}  # 桁数の組 → [行数, 異なる行の数]
for line, n in count.items():
    row = rows.setdefault(digit_pair(line), [0, 0])
    row[0] += n
    row[1] += 1
print("桁数の組   行数   異なる行   重複で消える行")
for key in sorted(rows):
    n_lines, n_unique = rows[key]
    print(f"{key}  {n_lines:6,}  {n_unique:8,}  {n_lines - n_unique:8,}")
total_unique = sum(u for _, u in rows.values())
print(f"合計     {len(lines):6,}  {total_unique:8,}"
      f"  {len(lines) - total_unique:8,}")
