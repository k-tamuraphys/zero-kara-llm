"""バッチ推論の効果と、静的バッチ・連続バッチの比較。"""

import csv
import random
import time
from collections import deque

import numpy as np
import torch

from batcher import Batcher, Job
from common import load
from engine import Engine, pick
from llmbook.corpus import DATA_DIR

model, tok = load("base")
torch.set_num_threads(1)  # 時間の測定を安定させるため、1 スレッドで動かす
val = np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16).astype(np.int64)


def prompt_at(i: int, n: int = 16) -> list[int]:
    return val[1000 + 50 * i:1000 + 50 * i + n].tolist()


# ---- (1) バッチサイズと、スループット・1 ステップの時間 ----
print("バッチサイズ B と生成速度"
      "（プロンプト 16 トークン、各 64 トークン生成）")
SIZES = [1, 2, 4, 8, 16, 32, 64]
step_sec = {B: float("inf") for B in SIZES}
for _ in range(6):  # 6 回測って、いちばん速かった回を採る
    for B in SIZES:
        engine, slots = Engine(model, n_slots=B), torch.arange(B)
        last = pick(engine.prefill(slots, [prompt_at(i) for i in range(B)]))
        t0 = time.perf_counter()
        for _ in range(64):
            last = pick(engine.forward(slots, last[:, None]))
        step_sec[B] = min(step_sec[B], (time.perf_counter() - t0) / 64)
rows = [(B, t * 1000, B / t) for B, t in step_sec.items()]
with open("ch19/out/batch_size.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["B", "step_ms", "tok_per_s", "speedup"])
    for B, ms, tps in rows:
        w.writerow([B, f"{ms:.3f}", f"{tps:.0f}", f"{tps / rows[0][2]:.2f}"])
        print(f"  B = {B:2d}: 1 ステップ {ms:5.2f} ミリ秒"
              f"  {tps:6.0f} トークン/秒"
              f"（B = 1 の {tps / rows[0][2]:4.1f} 倍）")


# ---- (2) 静的バッチと連続バッチ ----
def make_jobs(n: int, seed: int) -> list[Job]:
    """短い仕事が多く、ときどき長い仕事が混ざる 64 件。"""
    rng = random.Random(seed)
    lengths = [rng.choice([4, 4, 8, 8, 16, 96]) for _ in range(n)]
    return [Job(prompt_at(i), max_tokens=m) for i, m in enumerate(lengths)]


def run(policy: str, n_slots: int = 8) -> dict:
    jobs = make_jobs(64, seed=0)
    finish, t0 = {}, time.perf_counter()
    for j in jobs:  # 終わった時刻を記録する
        j.on_token = (lambda tok_id, done, j=j:
                      finish.__setitem__(id(j), time.perf_counter() - t0)
                      if done else None)
    batcher = Batcher(Engine(model, n_slots), stop_id=-1)  # 途中で止めない
    queue, n_steps, busy = deque(jobs), 0, 0
    # [snippet: policies]
    while queue or batcher.running:
        if policy == "static":
            # 静的バッチ: 全スロットが空くまで、次の組を入れない
            n_new = min(len(queue), n_slots) if not batcher.running else 0
        else:
            # 連続バッチ: スロットが一つでも空いたら、すぐ次を入れる
            n_new = min(len(queue), len(batcher.free))
        if n_new > 0:
            batcher.admit([queue.popleft() for _ in range(n_new)])
        if batcher.running:
            busy += len(batcher.running)  # このステップで働くスロットの数
            n_steps += 1
            batcher.step()
    # [end: policies]
    total = time.perf_counter() - t0
    n_tokens = sum(len(j.tokens) for j in jobs)
    waits = np.array([finish[id(j)] for j in jobs])
    short = np.array([finish[id(j)] for j in jobs if j.max_tokens <= 16])
    return {"steps": n_steps, "util": busy / (n_steps * n_slots),
            "total": total, "tps": n_tokens / total, "n_tokens": n_tokens,
            "mean": waits.mean(), "short": short.mean()}


print("\n64 件の仕事（4〜96 トークン）を 8 スロットで処理する")
res = {}
for _ in range(5):  # 交互に 5 回ずつ測り、それぞれ最も速かった回を採る
    for p in ["static", "continuous"]:
        r = run(p)
        if p not in res or r["total"] < res[p]["total"]:
            res[p] = r
for name, p in [("静的バッチ", "static"), ("連続バッチ", "continuous")]:
    r = res[p]
    print(f"  {name}: {r['steps']:3d} ステップ"
          f"  スロットの稼働率 {r['util']:5.1%}"
          f"  合計 {r['total']:5.2f} 秒  {r['tps']:5.0f} トークン/秒")
    print(f"              完了までの平均 {r['mean']:5.2f} 秒"
          f"（16 トークン以下の仕事だけなら {r['short']:5.2f} 秒）")
s, c = res["static"], res["continuous"]
print(f"  連続バッチは、合計時間が {s['total'] / c['total']:.2f} 分の 1、"
      f"短い仕事の待ち時間が {s['short'] / c['short']:.1f} 分の 1")
print(f"  （生成したトークンはどちらも {c['n_tokens']} 個）")
