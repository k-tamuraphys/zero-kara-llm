"""第 15 章のスイープ: 入力を変えると見積もりがどう動くか。

使い方（code/ で）: python ch15/sweep_infra.py > ch15/out/sweep_infra.txt
csv は ch15/out/sweep_*.csv に書く。計算だけなので 1 秒で終わる。
(1) 書き込みの速さ（全体）と、保存で失う時間の割合
(2) 推論: 同時に処理する本数と、1 本あたりの速さの上限
(3) 事例の 3 案: MFU と日数
(4) GPU の枚数と、1 週間で学習できるモデルの大きさ
(5) 勾配の同期 2S/β（NVLink は片方向 450 GB/s で計算）
"""
import csv
import math
from pathlib import Path

from ch15.estimate import (GPUS, ckpt_bytes, plan, waste, write_seconds,
                           young_interval)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def save(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def logspace(lo, hi, n):
    """lo から hi まで、対数で等間隔に n 点。"""
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


# ---- (1) 書き込みの速さ（全体）を 2 GB/s から 2,000 GB/s まで変える ----
print("== (1) 書き込みの速さと、最適な間隔での損失（1 次近似）==")
sizes = {"70B": ckpt_bytes(70e9), "405B": ckpt_bytes(405e9)}
mtbfs = {"3.1h": 3.1 * 3600, "7.9h": 7.9 * 3600}
rows = []
for gbps in logspace(2, 2000, 31):
    row = [f"{gbps:.3f}"]
    for s in sizes.values():
        for t in mtbfs.values():
            c = write_seconds(s, gbps)
            tau = young_interval(c, t)
            row.append(f"{100 * waste(tau, c, t):.3f}")
    rows.append(row)
save("sweep_ckpt.csv",
     ["gbps", "w70_31", "w70_79", "w405_31", "w405_79"], rows)
for gbps in (2, 20, 250, 2000):
    parts = []
    for name, s in sizes.items():
        c = write_seconds(s, gbps)
        t = mtbfs["3.1h"]
        tau = young_interval(c, t)
        parts.append(f"{name}: 保存 {c:7.1f} 秒、間隔 {tau / 60:5.1f} 分、"
                     f"損失 {waste(tau, c, t):5.1%}")
    print(f"全体 {gbps:5d} GB/s（故障 3.1 時間ごと）| " + " | ".join(parts))

# ---- (2) 推論: 同時の本数 n と、1 本あたりの速さの上限 ----
# 1 ステップで、重み（16.1 GB）と、載っている全系列の KV キャッシュを読む。
print("== (2) Llama 3 8B を H100 1 枚で: 本数と速さの上限 ==")
bw = 3.35e12                       # H100 のメモリ帯域（バイト/秒）
w = 2 * 8.03e9                     # BF16 の重み（バイト）
free = GPUS["H100"][1] * 1e9 - w   # 重みを引いた残りのメモリ
for ctx in (8192, 2048):
    kv = 128 * 1024 * ctx          # 1 本の KV キャッシュ（バイト）
    n_max = int(free // kv)        # メモリに載る本数
    rows = []
    for n in range(1, n_max + 1):
        steps = bw / (w + n * kv)  # 1 秒に回せるステップ数の上限
        rows.append([n, f"{steps:.2f}", f"{n * steps:.1f}"])
    save(f"sweep_infer_{ctx // 1024}k.csv", ["n", "per_seq", "total"], rows)
    for n in (1, 10, 50, n_max):
        steps = bw / (w + n * kv)
        read_gb = (w + n * kv) / 1e9
        print(f"文脈 {ctx:5d}、{n:3d} 本: 1 ステップで {read_gb:5.1f}"
              f" GB 読む → 1 本 {steps:6.1f} トークン/秒、"
              f"全体 {n * steps:7.0f} トークン/秒")

# ---- (3) 事例の 3 案: MFU を 20% から 60% まで変える ----
print("== (3) 事例の案と MFU（日数 / 費用）==")
cases = {"A": (20e9, "H100", 1), "B": (100e9, "H100", 4),
         "B8": (100e9, "H100", 8), "C": (100e9, "B200", 2)}
rows = []
for k in range(20, 61, 2):
    mfu = k / 100
    row = [k]
    for d, gpu, g in cases.values():
        row.append(f"{plan(1e9, d, gpu, g, mfu)['days']:.3f}")
    rows.append(row)
save("sweep_mfu.csv", ["mfu", "A", "B", "B8", "C"], rows)
for k in (25, 40, 55):
    parts = []
    for name, (d, gpu, g) in cases.items():
        p = plan(1e9, d, gpu, g, k / 100)
        parts.append(f"{name} {p['days']:4.1f} 日 {p['cost']:6,.0f} ドル")
    print(f"MFU {k}%: " + " | ".join(parts))

# ---- (4) GPU の枚数と、期間内に学習できる N（D=20N、H100、MFU 40%）----
print("== (4) 枚数と、学習できる N（D=20N）==")
rows = []
for g in logspace(1, 16000, 41):
    c = GPUS["H100"][0] * 1e12 * 0.4 * g * 86400
    rows.append([f"{g:.2f}", f"{math.sqrt(c * 7 / 120) / 1e9:.3f}",
                 f"{math.sqrt(c * 30 / 120) / 1e9:.3f}"])
save("sweep_week.csv", ["g", "week_b", "month_b"], rows)
for g in (1, 4, 16, 64, 256, 1024):
    c = GPUS["H100"][0] * 1e12 * 0.4 * g * 7 * 86400
    print(f"H100 × {g:5d}: 1 週間で {math.sqrt(c / 120) / 1e9:6.1f}B")

# ---- (5) 勾配の同期 2S/β。β は片方向の帯域 ----
print("== (5) 同期 2S/β（秒）: 50 / 100 / 450 GB/s（片方向）==")
for name, n in (("1B", 1e9), ("Llama 3 8B", 8.03e9),
                ("Llama 3 70B", 70.55e9)):
    s = 2 * n
    t = [2 * s / (b * 1e9) for b in (50, 100, 450)]
    print(f"{name:12s} S = {s / 1e9:6.1f} GB: "
          + " / ".join(f"{x:.3f}" for x in t))
