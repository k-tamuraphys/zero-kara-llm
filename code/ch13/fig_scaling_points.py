"""scaling_summary.txt から、図に使う 3 点を csv に書き出す。

実行: python ch13/fig_scaling_points.py  （一瞬）
出力: ch13/out/scaling_points.csv（列は name, n_nonemb, val_loss）
"""

import re

rows = []
with open("ch13/out/scaling_summary.txt") as f:
    for line in f:
        m = re.match(r"(\w+): 全体 ([\d,]+) / 埋め込み以外 ([\d,]+)"
                     r" / 検証損失 ([\d.]+)", line)
        if m:
            rows.append((m[1], int(m[3].replace(",", "")), float(m[4])))
with open("ch13/out/scaling_points.csv", "w") as f:
    f.write("name,n_nonemb,val_loss\n")
    for name, n, loss in rows:
        f.write(f"{name},{n},{loss}\n")
        print(name, n, loss)
