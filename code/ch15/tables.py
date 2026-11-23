"""章の図表のもとになる数値を、電卓で計算して out/ に書き出す。

python ch15/tables.py
価格・仕様は research/hardware_infra.md（2026 年 11 月 22 日取得）。
"""
import csv
import math
from pathlib import Path

from ch15.estimate import (GPUS, ckpt_bytes, hms, infer_memory,
                           train_memory, waste, write_seconds,
                           young_interval)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
txt = []


def save(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---- GPU ごとの「1 EFLOP あたりの費用」（MFU 40%）----
# [snippet: eflop]
rows = []
for name, (tflops, mem, price) in GPUS.items():
    eflop_per_h = tflops * 1e12 * 0.4 * 3600 / 1e18
    rows.append((name, tflops, mem, price, f"{price / eflop_per_h:.2f}"))
# [end: eflop]
save("gpu_cost.csv", ["gpu", "tflops", "mem", "price", "usd_per_eflop"],
     rows)
txt.append("== 1 EFLOP（10^18 FLOPs）あたりの費用（MFU 40%）==")
txt += [f"{r[0]:<6} {r[1]:>5} TFLOP/s {r[2]:>4} GB {r[3]:>6.2f} ドル/時"
        f" → {r[4]} ドル/EFLOP" for r in rows]

# ---- H100 の 1 GPU 時間の価格（契約と事業者の種類）----
h100 = [("スポット（中央値）", 2.33), ("予約（中央値）", 3.28),
        ("GPU クラウド（中央値）", 3.99), ("Lambda", 4.59),
        ("CoreWeave", 6.16), ("AWS", 6.88), ("Azure", 6.98),
        ("Google Cloud", 11.07)]
save("h100_price.csv", ["label", "price"], h100)

# ---- 学習の状態メモリ（16 バイト/パラメータ）----
models = [("GPT-2 124M", 124e6), ("1B", 1e9), ("Llama 3 8B", 8.03e9),
          ("Llama 3 70B", 70.55e9), ("405B", 405e9)]
save("memory.csv", ["model", "train_gb", "infer_gb"],
     [(m, f"{train_memory(n) / 1e9:.1f}", f"{2 * n / 1e9:.1f}")
      for m, n in models])

# ---- チェックポイントの大きさと書き込み時間 ----
txt.append("== チェックポイント: 重みだけ(BF16) / 完全 / 書き込み時間 ==")
for m, n in [("GPT-2 124M", 124e6), ("1B", 1e9), ("7B", 7e9),
             ("70B", 70e9), ("405B", 405e9)]:
    full = ckpt_bytes(n)
    txt.append(f"{m:<12} {ckpt_bytes(n, False) / 1e9:>8.1f} GB "
               f"{full / 1e9:>8.1f} GB  2 GB/s: "
               f"{hms(write_seconds(full, 2)):>9}  20 GB/s: "
               f"{hms(write_seconds(full, 20)):>9}")

# ---- 保存の間隔と、失う時間の割合（70B、故障間隔 3.1 時間と 7.9 時間）----
c2, c20 = (write_seconds(ckpt_bytes(70e9), v) for v in (2, 20))
rows = []
for k in range(0, 61):
    tau = 60 * 2 * 10 ** (k / 30)       # 2 分から 200 分まで対数刻み
    rows.append((f"{tau / 60:.2f}",
                 f"{100 * waste(tau, c2, 3.1 * 3600):.2f}",
                 f"{100 * waste(tau, c20, 3.1 * 3600):.2f}",
                 f"{100 * waste(tau, c2, 7.9 * 3600):.2f}"))
save("ckpt_waste.csv", ["tau_min", "w2", "w20", "w2_m79"], rows)
txt.append("== 70B（840 GB）の保存の間隔の目安と、失う時間の割合 ==")
for label, c, m in (("2 GB/s, 故障間隔 3.1 時間", c2, 3.1),
                    ("20 GB/s, 故障間隔 3.1 時間", c20, 3.1),
                    ("2 GB/s, 故障間隔 7.9 時間", c2, 7.9)):
    tau = young_interval(c, m * 3600)
    txt.append(f"{label:<24}: 保存 {hms(c)}、間隔 {tau / 60:.0f} 分、"
               f"損失 {waste(tau, c, m * 3600):.1%}")

# ---- 勾配の同期時間 2S/β（S は BF16 の勾配 2N バイト）----
txt.append("== 勾配の同期 2S/β（秒）: 50 / 100 / 900 GB/s ==")
for m, n in models[1:4]:
    s = 2 * n
    txt.append(f"{m:<12} S = {s / 1e9:>6.1f} GB: " + " / ".join(
        f"{2 * s / (b * 1e9):.3f}" for b in (50, 100, 900)))
tps = GPUS["H100"][0] * 1e12 * 0.4 / (6 * 8.03e9)
txt.append(f"Llama 3 8B を H100 1 枚で: {tps:,.0f} トークン/秒、"
           f"16,384 トークンの 1 ステップは {16384 / tps:.2f} 秒")

# ---- 買うか借りるか（B300 8 枚のサーバ。概算）----
# [snippet: breakeven]
BUY = 550_000                           # 8 GPU サーバの購入額（概算）
RENT = 9.00 * 8                         # 同じ GPU 8 枚を借りる（ドル/時）
POWER = 14.3 * 1.3 * 0.10               # kW × PUE × ドル/kWh（ドル/時）
rows = []
for month in range(0, 49):
    h = month * 730
    cost = [BUY + POWER * h]            # 買う: 電気は 24 時間分
    cost += [RENT * h * u for u in (1.0, 0.5, 0.25)]   # 借りる: 稼働率 u
    rows.append([month] + [f"{v / 1e3:.1f}" for v in cost])
# [end: breakeven]
save("breakeven.csv", ["month", "buy", "rent100", "rent50", "rent25"], rows)
txt.append("== 買うか借りるか（B300 × 8、購入 55 万ドル、電気 "
           f"{POWER:.2f} ドル/時）==")
for u in (1.0, 0.5, 0.25):
    h = BUY / (RENT * u - POWER)
    txt.append(f"稼働率 {u:.0%}: 借りると {RENT * u:.1f} ドル/時 → "
               f"{h:,.0f} 時間（{h / 730:.1f} か月）で購入額に並ぶ")

# ---- 電気代: kW × PUE × 単価 ----
it_kw = 10.2
txt.append("== 電気代（8×H100 ノード 10.2 kW、PUE 1.3、0.10 ドル/kWh）==")
txt.append(f"1 時間 {it_kw * 1.3 * 0.10:.2f} ドル = 1 GPU 時間 "
           f"{it_kw * 1.3 * 0.10 / 8:.3f} ドル（借りる価格 3.99 ドルの "
           f"{it_kw * 1.3 * 0.10 / 8 / 3.99:.1%}）")
txt.append(f"IT 1 MW を 1 年: {1000 * 1.3 * 8760 / 1e6:.1f} GWh、"
           f"{1000 * 1.3 * 8760 * 0.10 / 1e6:.2f} 百万ドル")

# ---- 1 週間で学習できる最大のモデル（D = 20N、MFU 40%）----
txt.append("== 1 週間（と 30 日）で学習できる最大の N（D=20N, H100）==")
for k in (1, 8, 248, 1016, 16000):
    out = []
    for days in (7, 30):
        c = GPUS["H100"][0] * 1e12 * 0.4 * k * days * 86400
        out.append(f"{math.sqrt(c / 120) / 1e9:6.1f}B")
    txt.append(f"H100 × {k:>6,}: " + " / ".join(out)
               + f"  （電力 約 {k / 8 * 10.2:,.0f} kW）")

# ---- 推論のメモリ（Llama 3 8B、KV キャッシュ 128 KiB/トークン）----
txt.append("== 推論: Llama 3 8B（8.03B）、文脈 8,192、BF16 ==")
w = infer_memory(8.03e9, 128 * 1024, 8192, n_seq=0)
kv = 128 * 1024 * 8192
txt.append(f"重み {w / 1e9:.1f} GB、KV キャッシュ 1 本 {kv / 1e9:.2f} GB")
for gpu, mem in (("RTX 4090", 24), ("RTX 5090", 32), ("L40S", 48),
                 ("H100", 80), ("H200", 141)):
    n_seq = int((mem * 1e9 - w) // kv)
    txt.append(f"{gpu:<8} {mem:>3} GB: 同時に {n_seq:>3} 本")
(OUT / "tables.txt").write_text("\n".join(txt) + "\n")
print("\n".join(txt))
