"""学習前・コールドスタート後・RL 後の軌跡を並べ、RL 後の誤りを分類する。
  python examples.py > out/examples.txt"""
import csv, re
from env import OUT_DIR

runs = {"学習前": "before", "SFT後": "cold", "SFT+RL(c=0)": "cold_c0"}
rows = {k: [r for r in csv.DictReader(open(OUT_DIR / f"eval_{v}.csv"))
            if r["kind"] == "add"] for k, v in runs.items()}
for i in [0, 3, 5]:
    print(f"質問: {rows['学習前'][i]['question']}")
    for k in runs:
        r = rows[k][i]
        t = r["traj"].replace("<|end|>", "")
        print(f"  {k:12s}{'○' if r['correct'] == '1' else '×'} {t}")
wrong = [r for r in rows["SFT+RL(c=0)"] if r["correct"] == "0"]
copy = sum(1 for r in wrong if (m := re.search(r"<\|tool\|>(.*?)<\|end\|>", r["traj"]))
           and sorted(re.findall(r"\d+", m.group(1))) != sorted(re.findall(r"\d+", r["question"])[:2]))
print(f"SFT+RL(c=0) の加減算の誤り {len(wrong)} 問のうち、式の数を写し間違えたもの {copy} 問")
