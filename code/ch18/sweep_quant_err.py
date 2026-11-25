"""第 18 章: ビット数と区切り方を変えて、量子化の誤差を 3 通りに測る。

(1) モデル全体の重みの誤差の平均、(2) 重みの誤差の最大値、
(3) 計算の対話の答えの位置で、元のモデルとの出力の分布のずれ（KL）。
正答率は測らない（測った 7 通りは ch18/out/quant.csv にある）。
python ch18/sweep_quant_err.py > ch18/out/sweep_quant_err.txt
"""
import csv
import random

import torch
import torch.nn.functional as F

from llmbook.chat import IGNORE, encode_chat_example, make_chat_example
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

from quantize import dequantize, quantize, quantize_model

torch.set_num_threads(2)
OUT = DATA_DIR.parent / "ch18" / "out"
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "sft.pt").eval()

# 評価に使う対話（答えの位置だけを見る）
rng = random.Random(2024)
data = []
for _ in range(200):
    ids, labels = encode_chat_example(tok, make_chat_example(rng)[0])
    data.append((torch.tensor([ids]), torch.tensor(labels) != IGNORE))
with torch.no_grad():
    ref = [F.log_softmax(model(x)[0][0][m], -1) for x, m in data]

# quant.csv の正答率（測ってある設定だけ）
acc = {}
with open(OUT / "quant.csv") as f:
    for r in csv.DictReader(f):
        acc[r["method"]] = r["acc"]

rows = []
for group in (None, 32):
    for bits in (8, 7, 6, 5, 4, 3, 2):
        errs = []
        for p in model.parameters():
            if p.dim() == 2:
                w = p.data
                errs.append((dequantize(*quantize(w, bits, group)) - w)
                            .abs().flatten())
        e = torch.cat(errs)
        m, mb = quantize_model(model, bits, group)
        kls = []
        with torch.no_grad():
            for (x, mask), lp in zip(data, ref):
                lq = F.log_softmax(m(x)[0][0][mask], -1)
                kls.append((lp.exp() * (lp - lq)).sum(-1).mean().item())
        name = f"int{bits}-" + (f"g{group}" if group else "row")
        rows.append(dict(method=name, bits=bits,
                         group="g32" if group else "row",
                         mb=round(mb, 3), mean_err=e.mean().item(),
                         max_err=e.max().item(),
                         kl=sum(kls) / len(kls), acc=acc.get(name, "")))
        r = rows[-1]
        print(f"{name:9s} {mb:5.2f} MB  mean|err| {r['mean_err']:.6f}  "
              f"max|err| {r['max_err']:.4f}  KL {r['kl']:.5f}  "
              f"acc {r['acc'] or '-'}", flush=True)
# 図用に、全部・区切り方ごと・正答率のある設定だけ、に分けて書く
parts = {"": rows,
         "_row": [r for r in rows if r["group"] == "row"],
         "_g32": [r for r in rows if r["group"] == "g32"],
         "_acc": [r for r in rows if r["acc"] != ""],
         "_acc_row": [r for r in rows if r["acc"] != "" and
                      r["group"] == "row"],
         "_acc_g32": [r for r in rows if r["acc"] != "" and
                      r["group"] == "g32"]}
for suffix, part in parts.items():
    with open(OUT / f"sweep_quant_err{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(part)
