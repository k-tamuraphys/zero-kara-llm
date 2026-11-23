"""学習前後の評価（固定問題）を集計する。
  python analyze.py  → out/compare.txt, out/breakdown.csv
学習前の評価がなければ作る（eval_before.csv）。"""
import csv, re, sys
import torch
from env import *

torch.set_num_threads(1)
if not (OUT_DIR / "eval_before.csv").exists():
    res, trs = evaluate(load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt"), eval_sets())
    with open(OUT_DIR / "eval_before.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "question", "answer", "calls", "correct", "traj"])
        for k, ts in trs.items():
            for t in ts:
                w.writerow([k, t.task.question, t.task.answer, t.n_calls,
                            int(reward(t, 0) == 1.0), "".join(t.texts)])

# [snippet: carry]
def bucket(q):
    """加減算の問題を「大きいほうの桁数」と「繰り上がり／繰り下がりの有無」で分ける。"""
    a, b = map(int, re.findall(r"\d+", q)[:2])
    sub = "-" in q or "引" in q
    carry, x, y, borrow = False, a, b, 0
    while x or y:
        dx, dy = x % 10, y % 10
        if (dx - dy - borrow < 0) if sub else (dx + dy + borrow >= 10):
            carry = True
        borrow = int((dx - dy - borrow < 0) if sub else (dx + dy + borrow >= 10))
        x, y = x // 10, y // 10
    return f"{len(str(max(a, b)))}桁・{'繰り上がりあり' if carry else 'なし'}"
# [end: carry]

runs = [("before", "学習前"), ("c0", "RLのみc=0"), ("c02", "RLのみc=0.2"),
        ("cold", "SFT後"), ("cold_c0", "SFT+RLc=0"), ("cold_c02", "SFT+RLc=0.2"),
        ("cold_c035", "SFT+RLc=0.35"), ("cold_c05", "SFT+RLc=0.5")]
runs = [(n, l) for n, l in runs if (OUT_DIR / f"eval_{n}.csv").exists()]
rows = {n: list(csv.DictReader(open(OUT_DIR / f"eval_{n}.csv"))) for n, _ in runs}
out = open(OUT_DIR / "compare.txt", "w")
def p(*a): print(*a); print(*a, file=out)
p("固定の評価問題（貪欲）: 加減算 1,000 問 / 掛け算 300 問 / 掛けて足す 300 問")
p(f"{'モデル':14s}{'種類':8s}{'正答率':>8s}{'電卓使用率':>10s}{'平均呼び出し':>10s}")
for n, l in runs:
    for k in ["add", "mul", "two"]:
        rs = [r for r in rows[n] if r["kind"] == k]
        acc = sum(int(r["correct"]) for r in rs) / len(rs)
        tool = sum(int(r["calls"]) > 0 for r in rs) / len(rs)
        calls = sum(int(r["calls"]) for r in rs) / len(rs)
        p(f"{l:14s}{k:8s}{acc:8.3f}{tool:10.3f}{calls:10.2f}")
p("\n加減算を桁数と繰り上がりで分けた電卓使用率（正答率）")
keys = sorted({bucket(r["question"]) for r in rows["before"] if r["kind"] == "add"})
with open(OUT_DIR / "breakdown.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bucket", "n"] + [f"{n}_{s}" for n, _ in runs for s in ("tool", "acc")])
    for b in keys:
        line, vals = f"{b:16s}", []
        for n, l in runs:
            rs = [r for r in rows[n] if r["kind"] == "add" and bucket(r["question"]) == b]
            tool = sum(int(r["calls"]) > 0 for r in rs) / len(rs)
            acc = sum(int(r["correct"]) for r in rs) / len(rs)
            line += f"  {l}: {tool:4.2f}（{acc:4.2f}）"
            vals += [round(tool, 3), round(acc, 3)]
        w.writerow([b, len(rs)] + vals)
        p(f"{line}  [{len(rs)} 問]")

# 学習曲線を 10 ステップの移動平均にする（図用）
for n in ["c0", "c02", "cold_c0", "cold_c02", "cold_c035", "cold_c05"]:
    f = OUT_DIR / f"train_{n}.csv"
    if not f.exists():
        continue
    rs = list(csv.DictReader(open(f)))
    with open(OUT_DIR / f"curve_{n}.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["step", "reward", "add_acc", "add_tool", "mul_acc", "kl"])
        for i in range(9, len(rs)):
            win = rs[i - 9:i + 1]
            avg = lambda k: round(sum(float(r[k]) for r in win) / 10, 4)
            w.writerow([rs[i]["step"], avg("reward"), avg("add_acc"),
                        avg("add_tool"), avg("mul_acc"), avg("kl")])
