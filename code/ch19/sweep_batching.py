"""静的バッチと連続バッチ処理を、スロットの数を変えて比べる（スイープ）。

batch_generate.py の (2) と同じ 64 件の仕事（4〜96 トークン）を、
スロット 1〜32 個で処理し、ステップ数・稼働率・かかった時間を記録する。
スロット 8 個のときは、どのスロットがいつどの仕事をしていたかも残す。
"""

import csv
import random
import time
from collections import deque

import numpy as np
import torch

from batcher import Batcher, Job
from common import load
from engine import Engine
from llmbook.corpus import DATA_DIR

model, tok = load("base")
torch.set_num_threads(1)  # batch_generate.py と同じく 1 スレッドで測る
torch.manual_seed(0)
val = np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16).astype(np.int64)
SLOTS = [1, 2, 4, 8, 16, 32]


def prompt_at(i: int, n: int = 16) -> list[int]:
    return val[1000 + 50 * i:1000 + 50 * i + n].tolist()


def make_jobs(n: int, seed: int) -> list[Job]:
    """batch_generate.py と同じ 64 件（短い仕事が多く、長い仕事が混ざる）。"""
    rng = random.Random(seed)
    lengths = [rng.choice([4, 4, 8, 8, 16, 96]) for _ in range(n)]
    return [Job(prompt_at(i), max_tokens=m) for i, m in enumerate(lengths)]


def run(policy: str, n_slots: int) -> dict:
    """64 件を処理し、ステップ数・稼働率・時間と、スロットの使われ方を返す。"""
    jobs = make_jobs(64, seed=0)
    batcher = Batcher(Engine(model, n_slots), stop_id=-1)  # 途中で止めない
    queue, n_steps, busy = deque(jobs), 0, 0
    spans = {}  # 仕事の番号 -> [スロット, 始めたステップ, 終えたステップ]
    t0 = time.perf_counter()
    while queue or batcher.running:
        if policy == "static":
            n_new = min(len(queue), n_slots) if not batcher.running else 0
        else:
            n_new = min(len(queue), len(batcher.free))
        if n_new > 0:
            batcher.admit([queue.popleft() for _ in range(n_new)])
        if batcher.running:
            for slot, job in batcher.running.items():
                k = jobs.index(job)
                spans.setdefault(k, [slot, n_steps, n_steps])
                spans[k][2] = n_steps + 1
            busy += len(batcher.running)
            n_steps += 1
            batcher.step()
    total = time.perf_counter() - t0
    return {"steps": n_steps, "util": busy / (n_steps * n_slots),
            "total": total, "spans": spans,
            "long": [j.max_tokens > 16 for j in jobs]}


rows, gantt = [], {}
for n_slots in SLOTS:
    for policy in ["static", "continuous"]:
        best = min((run(policy, n_slots) for _ in range(3)),
                   key=lambda r: r["total"])  # 3 回のうち最も速かった回
        rows.append((n_slots, policy, best))
        if n_slots == 8:
            gantt[policy] = best
        print(f"スロット {n_slots:2d}  {policy:10s}"
              f"  {best['steps']:4d} ステップ"
              f"  稼働率 {best['util']:5.1%}  合計 {best['total']:5.2f} 秒")

with open("ch19/out/sweep_batching.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["slots", "static_steps", "cont_steps", "static_util",
                "cont_util", "static_s", "cont_s"])
    for n_slots in SLOTS:
        s, c = [r for k, p, r in rows if k == n_slots]
        w.writerow([n_slots, s["steps"], c["steps"],
                    f"{100 * s['util']:.1f}", f"{100 * c['util']:.1f}",
                    f"{s['total']:.3f}", f"{c['total']:.3f}"])

# スロット 8 個のときの使われ方（図の帯。方法ごとに 1 つのファイル）
for policy, r in gantt.items():
    with open(f"ch19/out/sweep_batching_gantt_{policy}.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "slot", "start", "len", "long"])
        for k, (slot, a, b) in sorted(r["spans"].items()):
            w.writerow([k, slot, a, b - a, int(r["long"][k])])
