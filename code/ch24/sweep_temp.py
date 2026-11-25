"""温度と、最初の役割トークンの強制を変えて、加減算 1,000 問の正答率を測る。
  python ch24/sweep_temp.py > ch24/out/sweep_temp.txt  → out/sweep_temp.csv
モデル: コールドスタート後（RL の出発点）、SFT+RL c=0、SFT+RL c=0.5。
条件: 貪欲／温度 1 × 「モデルに任せる」「電卓に強制」「暗算に強制」、
      温度 1 で探索 eps=0.2（学習中のロールアウトと同じ条件）。
学習はしない。評価の問題（eval_sets）を使う。
"""
import csv
import re
import time

import torch
import env
from env import *

torch.set_num_threads(2)
t0 = time.time()
_orig_first = env._sample_first


def forced_first(role):
    """最初のターンの最初のトークンを role に固定する _sample_first。
    rollout は最初のターンだけ eps を渡すので、eps>0 のときだけ固定する。"""
    def f(logits, temperature, eps, gen):
        nxt = _orig_first(logits, temperature, 0.0, gen)
        if eps > 0:
            nxt = torch.full_like(nxt, role)
        return nxt
    return f


def bucket(q):
    """analyze.py と同じ分け方（大きいほうの桁数と、繰り上がりの有無）。"""
    a, b = map(int, re.findall(r"\d+", q)[:2])
    sub = "-" in q or "引" in q
    carry, x, y, borrow = False, a, b, 0
    while x or y:
        dx, dy = x % 10, y % 10
        if (dx - dy - borrow < 0) if sub else (dx + dy + borrow >= 10):
            carry = True
        borrow = int((dx - dy - borrow < 0) if sub
                     else (dx + dy + borrow >= 10))
        x, y = x // 10, y // 10
    return f"{len(str(max(a, b)))}桁・{'あり' if carry else 'なし'}"


tasks = eval_sets()["add"]
keys = sorted({bucket(t.question) for t in tasks})
# (条件の名前, 温度, eps, 最初の役割の固定)
conds = [("greedy", 0.0, 0.0, None), ("t1", 1.0, 0.0, None),
         ("t1_eps", 1.0, 0.2, None),
         ("greedy_tool", 0.0, 1.0, TOOL), ("greedy_asst", 0.0, 1.0, ASSIST),
         ("t1_tool", 1.0, 1.0, TOOL), ("t1_asst", 1.0, 1.0, ASSIST)]
models = [("cold", "コールドスタート後"), ("cold_c0", "SFT+RL c=0"),
          ("cold_c05", "SFT+RL c=0.5")]
rows = []
for name, label in models:
    model = load_checkpoint(CKPT_DIR / f"ch24_{name}.pt")
    for cname, temp, eps, role in conds:
        if name != "cold_c0" and cname == "t1_eps":
            continue                       # 学習中の条件は c=0 だけで測る
        env._sample_first = forced_first(role) if role else _orig_first
        gen = torch.Generator().manual_seed(1)   # 学習とは別の種
        trs = []
        for s in range(0, len(tasks), 250):
            trs += rollout(model, tasks[s:s + 250], temperature=temp,
                           eps=eps, gen=gen)
        env._sample_first = _orig_first
        ok = [reward(t, 0) == 1.0 for t in trs]
        tool = [t.n_calls > 0 for t in trs]
        row = {"model": name, "cond": cname, "all": sum(ok) / len(ok),
               "tool": sum(tool) / len(tool)}
        for k in keys:
            ix = [i for i, t in enumerate(tasks) if bucket(t.question) == k]
            row[k] = sum(ok[i] for i in ix) / len(ix)
        rows.append(row)
        print(f"{label:16s} {cname:12s} 正答率 {row['all']:.3f}"
              f"  電卓使用率 {row['tool']:.3f}  "
              + "  ".join(f"{k} {row[k]:.2f}" for k in keys), flush=True)

with open(OUT_DIR / "sweep_temp.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "cond", "all", "tool"] + keys)
    for row in rows:
        w.writerow([row["model"], row["cond"], round(row["all"], 4),
                    round(row["tool"], 4)] + [round(row[k], 4) for k in keys])
print("問題数: " + "  ".join(
    f"{k} {sum(bucket(t.question) == k for t in tasks)}" for k in keys))
print(f"[{time.time() - t0:.0f} 秒]")
