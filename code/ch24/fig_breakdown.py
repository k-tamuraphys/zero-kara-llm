"""第 24 章の 2 つのグラフのための表を作る（集計だけ。学習も生成もしない）。
  python ch24/fig_breakdown.py
入力: out/breakdown.csv（analyze.py）、out/sweep_temp.csv（sweep_temp.py）
出力:
  out/breakdown_by_c.csv  コスト c ごとの、問題の種類別の電卓使用率と正答率
                          （SFT+RL。c を横軸に描くため、縦横を入れ替える）
  out/fig_threshold_<model>.csv  温度 1 で測った暗算の正答率 p と、
                          電卓の期待値 p'-c（SFT+RL c=0 と c=0.5）
"""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
rows = list(csv.DictReader(open(OUT / "breakdown.csv")))
# 種類の名前を短い列名にする（pgfplots の列名に日本語を使わない）
short = {"1桁・なし": "d1n", "1桁・繰り上がりあり": "d1c",
         "2桁・なし": "d2n", "2桁・繰り上がりあり": "d2c",
         "3桁・なし": "d3n", "3桁・繰り上がりあり": "d3c"}
runs = [(0.0, "cold_c0"), (0.2, "cold_c02"), (0.35, "cold_c035"),
        (0.5, "cold_c05")]
with open(OUT / "breakdown_by_c.csv", "w", newline="") as f:
    w = csv.writer(f)
    names = [short[r["bucket"]] for r in rows]
    w.writerow(["c"] + [f"{n}_tool" for n in names]
               + [f"{n}_acc" for n in names])
    for c, run in runs:
        w.writerow([c] + [r[f"{run}_tool"] for r in rows]
                   + [r[f"{run}_acc"] for r in rows])

# sweep_temp.csv の種類の名前（「あり」）を breakdown.csv の名前に合わせる
temp = list(csv.DictReader(open(OUT / "sweep_temp.csv")))
def get(model, cond):
    return next(r for r in temp if r["model"] == model and r["cond"] == cond)
for model, c, run in [("cold_c0", 0.0, "cold_c0"),
                      ("cold_c05", 0.5, "cold_c05")]:
    asst, tool = get(model, "t1_asst"), get(model, "t1_tool")
    with open(OUT / f"fig_threshold_{model}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "p", "pc", "tool", "lab"])
        for r in rows:
            k = r["bucket"].replace("繰り上がりあり", "あり")
            p, pp = float(asst[k]), float(tool[k])
            u = f"{float(r[f'{run}_tool']):.2f}"       # 電卓使用率（貪欲）
            lab = r["bucket"].replace("繰り上がり", "").replace("・", "") \
                + " " + u                               # 図の点に添える字
            w.writerow([short[r["bucket"]], round(p, 3), round(pp - c, 3),
                        u, lab])
            print(f"{model:9s} {r['bucket']:12s} 暗算 p={p:.2f}  "
                  f"電卓 p'-c={pp - c:+.2f}  "
                  f"電卓使用率（貪欲）{r[f'{run}_tool']}")
