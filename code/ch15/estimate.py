"""見積もり電卓：パラメータ数から GPU 時間・費用・メモリ・保存容量まで。

使い方（code/ で）: python ch15/estimate.py
ほかのスクリプトからは from ch15.estimate import plan のように使う。
数値の出典は research/hardware_infra.md（2026 年 11 月 22 日取得）。
"""
import math

# [snippet: gpus]
# 名前: (密な BF16 の公称 TFLOP/s, メモリ GB, ドル/GPU 時の中央値)
GPUS = {
    "A100": (312, 80, 2.00),
    "L40S": (362, 48, 1.62),
    "H100": (989, 80, 3.99),
    "H200": (989, 141, 4.90),
    "B200": (2250, 180, 8.31),
    "GB200": (2500, 186, 10.50),
}
# [end: gpus]


# [snippet: train]
def train_flops(n_params, n_tokens):
    """学習の総計算量 C ≈ 6ND。"""
    return 6 * n_params * n_tokens


def gpu_hours(flops, gpu="H100", mfu=0.4):
    """実効 FLOP/s（公称 × MFU）で割って、GPU 時間にする。"""
    peak = GPUS[gpu][0] * 1e12
    return flops / (peak * mfu) / 3600


def plan(n_params, n_tokens, gpu="H100", n_gpu=8, mfu=0.4, price=None):
    """台数を決めたときの GPU 時間・日数・費用。"""
    c = train_flops(n_params, n_tokens)
    gh = gpu_hours(c, gpu, mfu)
    price = GPUS[gpu][2] if price is None else price
    return {"flops": c, "gpu_hours": gh,
            "days": gh / n_gpu / 24, "cost": gh * price}


def tokens_per_s(n_params, gpu="H100", n_gpu=8, mfu=0.4):
    """学習が 1 秒に処理するトークン数。"""
    return GPUS[gpu][0] * 1e12 * mfu * n_gpu / (6 * n_params)
# [end: train]


# [snippet: memory]
def train_memory(n_params, act_bytes=0):
    """学習時: 状態 16 バイト/パラメータ + 活性化。"""
    return 16 * n_params + act_bytes


def infer_memory(n_params, kv_per_token, ctx, n_seq=1, w_bytes=2):
    """推論時: 重み（BF16 で 2 バイト）+ KV キャッシュ × 同時の本数。"""
    return w_bytes * n_params + kv_per_token * ctx * n_seq
# [end: memory]


# [snippet: storage]
def data_bytes(n_tokens, bytes_per_token=2):
    """トークン列のバイナリ。uint16 なら 2 バイト/トークン。"""
    return n_tokens * bytes_per_token


def ckpt_bytes(n_params, full=True):
    """full: FP32 の重み + AdamW の m, v で 12 バイト/パラメータ。
    重みだけを BF16 で配るなら 2 バイト/パラメータ。"""
    return (12 if full else 2) * n_params


def write_seconds(n_bytes, gb_per_s):
    """書き込みにかかる秒数（速度は GB/s）。"""
    return n_bytes / (gb_per_s * 1e9)


def young_interval(ckpt_s, mtbf_s):
    """保存の間隔の目安 sqrt(2 c T_fail)（Young の式）。"""
    return math.sqrt(2 * ckpt_s * mtbf_s)


def waste(interval_s, ckpt_s, mtbf_s):
    """保存と、故障によるやり直しで失う時間の割合（1 次近似）。"""
    return ckpt_s / interval_s + interval_s / (2 * mtbf_s)
# [end: storage]


def si(x, unit=""):
    """1.23e9 -> '1.23 G' のように読みやすくする。"""
    for p, name in ((1e24, "Y"), (1e21, "Z"), (1e18, "E"), (1e15, "P"),
                    (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= p:
            return f"{x / p:.3g} {name}{unit}"
    return f"{x:.3g} {unit}"


def hms(sec):
    """秒を「日・時間・分・秒」のうち読みやすい単位にする。"""
    if sec >= 2 * 86400:
        return f"{sec / 86400:.1f} 日"
    if sec >= 2 * 3600:
        return f"{sec / 3600:.1f} 時間"
    if sec >= 120:
        return f"{sec / 60:.1f} 分"
    return f"{sec:.1f} 秒"


def pad(text, width, left=False):
    """全角文字を 2 桁と数えて、表示幅 width にそろえる。"""
    n = sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)
    fill = " " * max(0, width - n)
    return text + fill if left else fill + text


# [snippet: sheet]
def sheet(name, n_params, n_tokens, gpu, n_gpu, mfu=0.4, price=None,
          write_gbps=2.0, mtbf_h=None, bytes_per_token=2):
    """見積もりシートを 1 枚、文字列で返す。"""
    p = plan(n_params, n_tokens, gpu, n_gpu, mfu, price)
    tps = tokens_per_s(n_params, gpu, n_gpu, mfu)
    ck = ckpt_bytes(n_params)
    ck_s = write_seconds(ck, write_gbps)
    rows = [
        f"== {name} ==",
        f"N = {n_params:.3g}, D = {n_tokens:.3g}, D/N = "
        f"{n_tokens / n_params:.0f}",
        f"計算量 6ND         : {p['flops']:.3g} FLOPs",
        f"GPU 時間           : {p['gpu_hours']:,.0f}"
        f"（{gpu}, MFU {mfu:.0%}）",
        f"日数               : {n_gpu} 枚で {hms(p['days'] * 86400)}",
        f"費用               : {p['cost']:,.0f} ドル",
        f"学習の状態メモリ   : {si(train_memory(n_params), 'B')}"
        f"（{gpu} は 1 枚 {GPUS[gpu][1]} GB）",
        f"トークン列の容量   : "
        f"{si(data_bytes(n_tokens, bytes_per_token), 'B')}",
        f"読み出しの必要速度 : {si(tps, 'トークン/秒')} = "
        f"{si(tps * bytes_per_token, 'B/s')}",
        f"チェックポイント   : {si(ck, 'B')}、"
        f"{write_gbps:g} GB/s で {hms(ck_s)}",
    ]
    if mtbf_h is not None:
        tau = young_interval(ck_s, mtbf_h * 3600)
        rows.append(f"保存の間隔の目安   : {hms(tau)}ごと（故障間隔 "
                    f"{hms(mtbf_h * 3600)}、損失 "
                    f"{waste(tau, ck_s, mtbf_h * 3600):.1%}）")
    return "\n".join(rows)
# [end: sheet]


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    # 例: 80 億パラメータを、その 20 倍のトークンで、H100 64 枚で学習
    text = sheet("8B を 1,600 億トークン、H100 × 64", 8e9, 160e9,
                 "H100", 64)
    (out / "estimate.txt").write_text(text + "\n")
    print(text)
