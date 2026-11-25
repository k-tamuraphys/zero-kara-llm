"""第 8 章: 図（主成分分析の散布図）に名前を書き込む点だけを選んだ表を作る。

実行: cd code && python ch08/fig_pca_labels.py
      （embedding_neighbors.py が書いた pca_*.csv を読むだけ。数秒）
出力: ch08/out/pca_labels.csv
"""
import csv

from capture import OUT

# 名前を書き込むトークン（本文で話題にするもの）
PICK = ["7", "0", "三", "一", "先生", "奥さん", "私", "です。"]
# 「は」は点が混み合う所にあるので、本文の図で引き出し線を付けて書く
# 名前を点の左に書くもの（右に書くと、ほかの点や名前と重なる）
LEFT = ["7", "0", "先生"]
rows = []
for g in ["digit", "kanji", "person", "ending", "particle"]:
    with open(OUT / f"pca_{g}.csv", encoding="utf-8") as f:
        rows += [r for r in csv.DictReader(f) if r["label"] in PICK]
with open(OUT / "pca_labels.csv", "w", encoding="utf-8") as f:
    f.write("x,y,label,anchor\n")
    for r in rows:
        anchor = "east" if r["label"] in LEFT else "west"
        f.write(f"{r['x']},{r['y']},{r['label']},{anchor}\n")
print(f"{len(rows)} 個の点に名前を付ける")
