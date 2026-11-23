"""ケーススタディ:「10 億パラメータのモデルを 1 週間で学習したい」。

電卓で 3 つの案を比べる。python ch15/case_1b.py
"""
from pathlib import Path

from ch15.estimate import pad, plan, sheet

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# [snippet: case]
N = 1e9
cases = [  # 名前, トークン数, GPU, 枚数
    ("案 A: D=20N, H100 × 1", 20 * N, "H100", 1),
    ("案 B: D=100N, H100 × 4", 100 * N, "H100", 4),
    ("案 C: D=100N, B200 × 2", 100 * N, "B200", 2),
]
plans = [plan(N, d, gpu, k) for _, d, gpu, k in cases]
slow = [plan(N, d, gpu, k, mfu=0.25) for _, d, gpu, k in cases]
# [end: case]
out = [pad("", 24) + pad("GPU 時間", 9) + pad("日数", 9) + pad("費用", 12)
       + pad("MFU 25% なら", 22)]
for (name, *_), p, q in zip(cases, plans, slow):
    out.append(pad(name, 24, True) + f"{p['gpu_hours']:>9,.0f}"
               + f"{p['days']:>6.1f} 日{p['cost']:>7,.0f} ドル"
               + f"{q['days']:>7.1f} 日{q['cost']:>7,.0f} ドル")
(OUT / "case_1b.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
name, d, gpu, k = cases[1]
text = sheet(name, N, d, gpu, k, mtbf_h=47.7 * 24)
(OUT / "case_sheet.txt").write_text(text + "\n")
print(text)
