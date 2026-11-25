"""phrasing.py の出力から、言い回しごとの内訳を積み上げ棒の csv にする。

内訳は 4 つ: 電卓を呼んで正解／呼んで不正解／呼ばずに正解／呼ばずに不正解。
モデルは動かさず、ch23/out/phrasing.txt の「呼んだ x 問中 y 問、
呼ばない z 問中 w 問が正解」の行を読むだけ。
実行:  python ch23/fig_phrasing.py      出力: ch23/out/phrasing_counts.csv
"""

import csv
import re

from common import OUT_DIR

pat = re.compile(r"呼んだ (\d+) 問中 (\d+) 問、呼ばない (\d+) 問中 (\d+) 問")
rows = []
for line in (OUT_DIR / "phrasing.txt").read_text().splitlines():
    m = pat.search(line)
    if m:
        n_call, ok_call, n_no, ok_no = map(int, m.groups())
        rows.append({"style": len(rows) + 1,
                     "call_ok": ok_call, "call_ng": n_call - ok_call,
                     "no_ok": ok_no, "no_ng": n_no - ok_no})
with open(OUT_DIR / "phrasing_counts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
