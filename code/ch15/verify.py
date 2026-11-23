"""電卓の検算: (1) 本書のミニGPT (2) GPT-2 124M (3) Llama 級。

見積もりを、実測値や公開値と比べる。python ch15/verify.py
"""
import csv
import os
from pathlib import Path

from ch15.estimate import (GPUS, ckpt_bytes, data_bytes, gpu_hours, hms,
                           pad, plan, train_flops)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
CODE = Path(__file__).resolve().parent.parent


def ratio(est, ref):
    return f"見積もり/実際 = {est / ref:.2f}"


# ---- (1) 本書のミニGPT（base.pt）。実測値は第 12・13 章と README ----
# [snippet: mini]
N, D = 2_583_936, 4000 * 4096          # パラメータ数、延べトークン数
GPUS["CPU"] = (0.1274, 0, 0.0)         # 2 スレッドの行列積 127.4 GFLOP/s
step_s = 0.731                         # 第 12 章で測った 1 ステップの秒数
mfu = 6 * N * 4096 / step_s / (GPUS["CPU"][0] * 1e12)
p = plan(N, D, "CPU", n_gpu=1, mfu=mfu)
sec = p["days"] * 86400
# [end: mini]
counted = 16_515_072 * D               # 第 12 章でフックで数えた FLOPs
base_pt = os.path.getsize(CODE / "checkpoints" / "base.pt")
train_bin = os.path.getsize(CODE / "data" / "train.bin")
lines = [
    "== (1) 本書のミニGPT（base.pt）==",
    f"計算量: 6ND = {p['flops']:.3g} / 数えた値 {counted:.3g} FLOPs"
    f" … {ratio(p['flops'], counted)}",
    f"MFU   : 6N×4096 トークン÷{step_s} 秒÷127.4 GFLOP/s = {mfu:.0%}",
    f"時間  : {hms(sec)}（{sec:,.0f} 秒）/ 実際 2,163 秒"
    f" … {ratio(sec, 2163)}",
    f"重み  : 4N = {4 * N:,} / base.pt {base_pt:,} バイト"
    f" … {ratio(4 * N, base_pt)}",
    f"データ: 2 バイト × 2,479,113 トークン = {data_bytes(2_479_113):,}"
    f" / train.bin {train_bin:,} バイト",
]
(OUT / "verify_mini.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- (2) GPT-2 124M を 100 億トークン（llm.c、8×A100、約 90 分）----
# [snippet: gpt2]
p = plan(124e6, 10e9, "A100", n_gpu=8, mfu=0.6, price=14 / 8)
minutes = p["days"] * 24 * 60
# [end: gpt2]
now = plan(124e6, 10e9, "A100", n_gpu=8, mfu=0.6)
lines = [
    "== (2) GPT-2 124M × 100 億トークン（8×A100, MFU 60%）==",
    f"計算量: 6ND = {p['flops']:.3g} FLOPs",
    f"時間  : {minutes:.0f} 分 / 報告 約 90 分 … {ratio(minutes, 90)}",
    f"費用  : {p['cost']:.1f} ドル（当時の 1.75 ドル/GPU 時）/ "
    f"報告 約 20 ドル … {ratio(p['cost'], 20)}",
    f"        {now['cost']:.1f} ドル（2026 年 11 月の中央値 "
    f"{GPUS['A100'][2]:.2f} ドル/GPU 時）",
    f"データ: {data_bytes(10e9) / 1e9:.0f} GB（uint16）/ 報告 約 19 GB",
    f"保存  : 完全なチェックポイント {ckpt_bytes(124e6) / 1e9:.2f} GB",
]
(OUT / "verify_gpt2.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- (3) Llama 級と公開事例。公開値は hardware_infra.md §7 ----
# [snippet: llama]
cases = [  # 名前, N, D, GPU, 枚数, 公開された日数
    ("GPT-2 124M (llm.c)", 124e6, 10e9, "A100", 8, 1.5 / 24),
    ("LLaMA 65B", 65e9, 1.4e12, "A100", 2048, 21),
    ("SmolLM3 3B", 3e9, 11.2e12, "H100", 384, 24),
]
rows = []
for name, n, d, gpu, k, days in cases:
    c = train_flops(n, d)
    pub = k * days * 24                 # 公開値から出した GPU 時間
    est = gpu_hours(c, gpu, mfu=0.4)    # MFU 40% と仮定した見積もり
    implied = gpu_hours(c, gpu, mfu=1.0) / pub   # 逆算した MFU
    rows.append((name, c, est, pub, est / pub, implied, d / n))
# [end: llama]
lines = ["== (3) 公開事例との比較（見積もりは MFU 40% を仮定）==",
         pad("モデル", 19, True) + pad("D/N", 6) + pad("6ND", 10)
         + pad("見積もり", 11) + pad("公開値", 11) + pad("比", 6)
         + pad("逆算MFU", 9)]
for name, c, est, pub, r, imp, dn in rows:
    lines.append(f"{name:<19}{dn:>6,.0f}{c:>10.3g}{est:>11,.0f}"
                 f"{pub:>11,.0f}{r:>6.2f}{imp:>9.0%}")
lines.append("（見積もり・公開値の単位は GPU 時間）")
lines.append("-- 計算量そのものが公開されている例 --")
for name, n, d, pub in [("GPT-3 175B", 175e9, 300e9, 3.14e23),
                        ("Llama 3.1 405B", 405e9, 15.6e12, 3.8e25)]:
    c = train_flops(n, d)
    lines.append(f"{name:<16} 6ND = {c:.3g} / 論文 {pub:.3g} FLOPs"
                 f" … {ratio(c, pub)}")
c = train_flops(405e9, 15.6e12)
d100 = gpu_hours(c, "H100", mfu=1.0) / 16000 / 24
lines.append(f"Llama 3.1 405B を H100 16,000 枚で: MFU 100% なら "
             f"{d100:.0f} 日、40% なら {d100 / 0.4:.0f} 日")
(OUT / "verify_llama.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

with open(OUT / "verify.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "flops", "est", "pub", "ratio", "mfu"])
    for name, c, est, pub, r, imp, dn in rows:
        w.writerow([name, f"{c:.4g}", f"{est:.1f}", f"{pub:.1f}",
                    f"{r:.3f}", f"{imp * 100:.1f}"])
