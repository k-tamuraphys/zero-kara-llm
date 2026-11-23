"""train_lora.py の出力を集めて、ランクの実験の表を CSV にする。"""

import re
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
rows = []
for r in [1, 2, 4, 8, 16, 64]:
    f = OUT / f"lora_r{r}_all.txt"
    if "検証損失" not in (f.read_text() if f.exists() else ""):
        continue
    s = f.read_text()
    n = int(re.search(r"パラメータ: ([\d,]+)", s).group(1).replace(",", ""))
    acc = float(re.search(r"正答率（1,000 問）: ([\d.]+)%", s).group(1))
    loss = float(re.search(r"検証損失: ([\d.]+)", s).group(1))
    rows.append(f"{r},{n},{acc},{loss}")
(OUT / "rank.csv").write_text("r,n_train,acc,soseki\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
