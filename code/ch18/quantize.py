"""第 18 章: 学習後量子化（線形量子化）を自作し、大きさと正答率を測る。

python ch18/quantize.py          # 量子化の方式・ビット数ごとの比較
python ch18/quantize.py prune    # 枝刈り（絶対値の小さい重みを 0 にする）
"""
import copy
import csv
import sys

import torch

from llmbook.chat import chat_accuracy
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint


# [snippet: quantize]
def quantize(w: torch.Tensor, bits: int, group: int | None = None):
    """非対称の線形量子化。group 個ずつ（None なら行ごと）にスケールを持つ。

    戻り値: 整数 q、スケール s、ゼロ点 zp。w ≈ s * (q - zp)。
    """
    shape = w.shape
    w = w.reshape(-1, group) if group else w.reshape(shape[0], -1)
    lo = w.min(dim=1, keepdim=True).values         # 各組の最小値
    hi = w.max(dim=1, keepdim=True).values         # 各組の最大値
    qmax = 2**bits - 1                             # 8 ビットなら 255
    s = (hi - lo).clamp(min=1e-8) / qmax           # 1 目盛りの幅
    zp = torch.round(-lo / s)                      # 0.0 に当たる整数
    q = torch.clamp(torch.round(w / s) + zp, 0, qmax)
    return q.to(torch.uint8), s, zp, shape


def dequantize(q, s, zp, shape) -> torch.Tensor:
    """整数から浮動小数点の近似値に戻す。"""
    return (s * (q.float() - zp)).reshape(shape)
# [end: quantize]


# [snippet: quantize-model]
def quantize_model(model, bits: int, group: int | None):
    """2 次元の重み（線形層と埋め込み）を量子化し、戻した値で置き換える。"""
    m = copy.deepcopy(model)
    n_bits = 0
    for name, p in m.named_parameters():
        if p.dim() == 2:
            q, s, zp, shape = quantize(p.data, bits, group)
            p.data = dequantize(q, s, zp, shape)
            n_bits += p.numel() * bits + s.numel() * 32  # s と zp で 32
        else:
            n_bits += p.numel() * 32                       # 正規化層はそのまま
    return m, n_bits / 8 / 1e6                             # MB
# [end: quantize-model]


def main():
    torch.set_num_threads(2)
    tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
    model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "sft.pt").eval()
    fp32_mb = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    n_eval = 1000
    if len(sys.argv) > 1 and sys.argv[1] == "prune":
        rows = []
        for frac in (0.0, 0.3, 0.5, 0.7, 0.9):
            m = copy.deepcopy(model)
            # [snippet: prune]
            for p in m.parameters():
                if p.dim() == 2:
                    k = int(p.numel() * frac)        # 0 にする個数
                    if k > 0:
                        th = p.abs().flatten().kthvalue(k).values
                        p.data[p.abs() <= th] = 0.0
            # [end: prune]
            acc = chat_accuracy(m, tok, n_eval, seed=777)
            print(f"prune {frac:.0%}: acc {acc:.3f}", flush=True)
            rows.append(dict(frac=frac, acc=acc))
        with open("ch18/out/prune.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frac", "acc"])
            w.writeheader()
            w.writerows(rows)
        return
    # 1 つの重み行列で、ビット数と誤差の関係を見る
    W = model.blocks[0].mlp.fc.weight.data
    with open("ch18/out/quant_hist.csv", "w") as f:   # 重みの分布（図用）
        h = torch.histc(W, bins=41, min=-0.205, max=0.205)
        f.write("center,count\n")
        for i, c in enumerate(h.tolist()):
            f.write(f"{-0.2 + 0.01 * i:.3f},{int(c)}\n")
    print(f"W: shape {tuple(W.shape)}, min {W.min():.4f}, max {W.max():.4f}")
    for bits in (8, 4, 3, 2):
        for group in (None, 32):
            err = (dequantize(*quantize(W, bits, group)) - W).abs().mean()
            print(f"bits={bits} group={group}: mean |error| {err:.6f}")
    print(f"\nfp32: {fp32_mb:.2f} MB", flush=True)
    rows = []
    acc = chat_accuracy(model, tok, n_eval, seed=777)
    rows.append(dict(method="fp32", bits=32, mb=round(fp32_mb, 3), acc=acc))
    print(f"fp32            {fp32_mb:6.2f} MB  acc {acc:.3f}", flush=True)
    for bits, group in ((8, None), (4, None), (4, 32), (3, None), (3, 32),
                        (2, 32)):
        m, mb = quantize_model(model, bits, group)
        acc = chat_accuracy(m, tok, n_eval, seed=777)
        name = f"int{bits}-" + (f"g{group}" if group else "row")
        rows.append(dict(method=name, bits=bits, mb=round(mb, 3), acc=acc))
        print(f"{name:14s}  {mb:6.2f} MB  acc {acc:.3f}", flush=True)
    with open("ch18/out/quant.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "bits", "mb", "acc"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
