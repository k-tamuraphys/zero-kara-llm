"""第 25 章：規模別の設定と、第 15 章の式による見積もりの表。

  python ch25/scale_config.py   # → ch25/out/scale_config.txt, scale_config.csv
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"

# [snippet: scale-estimate]
GPUS = {"H100": (989e12, 3.99)}  # 公称 BF16 FLOP/s、ドル/GPU 時間


def estimate(N, D, n_gpu, gpu="H100", mfu=0.4):
    """第 15 章の見積もりの手順: 計算量 → GPU 時間 → 日数・費用・メモリ。"""
    peak, price = GPUS[gpu]
    C = 6 * N * D                              # 学習の計算量（FLOPs）
    gpu_hours = C / (peak * mfu * 3600)        # 式(15.1) の GPU 時間
    days = gpu_hours / (n_gpu * 24)
    cost = gpu_hours * price
    mem_gb = 16 * N / 1e9                      # 重み・勾配・Adam の状態
    return C, gpu_hours, days, cost, mem_gb
# [end: scale-estimate]


# [snippet: scale-stages]
STAGES = [  # 名前, 層, 次元, N, D, GPU 枚数
    ("CPU（本書）", 4, 192, 2.6e6, 1.64e7, None),
    ("GPU 1 枚", 12, 768, 1.24e8, 2.5e9, 1),
    ("8 枚ノード", 24, 2048, 1.0e9, 2.0e10, 8),
    ("クラスタ", 32, 4096, 8.0e9, 1.6e11, 64),
]
# [end: scale-stages]

# [snippet: next-steps]
NEXT = [  # 名前, N, D, GPU 枚数, 1 パラメータあたりのバイト数
    ("(1) 1.24 億・D=20N", 1.24e8, 2.48e9, 1, 16),
    ("(1) 1.24 億・D=100N", 1.24e8, 1.24e10, 1, 16),
    ("(2) 80 億の LoRA", 8.0e9, 1.0e7, 1, 2),  # 重みは BF16 で固定
]
# [end: next-steps]


def main():
    OUT.mkdir(exist_ok=True)
    lines = [f"{'N':>8}{'D':>9}{'C(FLOPs)':>10}{'GPU-h':>8}{'days':>7}"
             f"{'cost($)':>9}{'mem(GB)':>8}  stage"]
    rows = ["stage,N,D,C,gpu_hours,days,cost,mem_gb"]
    for name, L, d, N, D, g in STAGES:
        C, gh, days, cost, mem = estimate(N, D, g or 1)
        if g is None:      # CPU は実測（第 13 章: 約 36 分）を載せる
            gh, days, cost = float("nan"), 36 / 60 / 24, 0.0
        ghs = "-" if g is None else f"{gh:.1f}"
        lines.append(f"{N:8.2g}{D:9.2g}{C:10.2g}{ghs:>8}{days:7.2f}"
                     f"{cost:9,.0f}{mem:8.2f}  {name}")
        rows.append(f"{name},{N},{D},{C},{gh:.1f},{days:.2f},{cost:.0f},{mem}")
    lines.append("")
    lines.append("次の一歩（H100、MFU 40%）")
    for name, N, D, g, b in NEXT:
        C, gh, days, cost, mem = estimate(N, D, g)
        mem = b * N / 1e9
        lines.append(f"{N:8.2g}{D:9.2g}{C:10.2g}{gh:8.2f}{days:7.2f}"
                     f"{cost:9,.1f}{mem:8.2f}  {name}")
    text = "\n".join(lines)
    print(text)
    (OUT / "scale_config.txt").write_text(text + "\n")
    (OUT / "scale_config.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
