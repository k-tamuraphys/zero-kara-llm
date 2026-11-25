"""load_100.py の出力（out/load_100.txt）から、図のための csv を作る。

測り直しはしない。同時 100 人のときの、スロットの数ごとの
最初の 1 個までの時間と、応答完了までの時間（中央値）を取り出す。
"""

import csv
import re

rows, slots = [], None
for line in open("ch19/out/load_100.txt", encoding="utf-8"):
    m = re.match(r"スロット (\d+) 個", line)
    if m:
        slots = int(m.group(1))
    m = re.search(r"同時 100 人:\s+(\d+) 件/秒.*?まで\s+([\d.]+) ミリ秒"
                  r".*?まで\s+([\d.]+) ミリ秒", line)
    if m:
        rows.append((slots, m.group(1), m.group(2), m.group(3)))
with open("ch19/out/load_100.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["slots", "rps", "ttft_ms", "total_ms"])
    w.writerows(rows)
print(rows)
