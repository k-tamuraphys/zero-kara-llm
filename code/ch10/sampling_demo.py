"""基盤モデル base.pt で、生成の設定（貪欲・温度・top-k・top-p）を比べる。

実行: python ch10/sampling_demo.py
出力: ch10/out/temperature.txt, temperature.csv, sampling.txt
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

OUT = Path(__file__).parent / "out"
torch.set_num_threads(2)
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt").eval()

# [snippet: temperature]
idx = torch.tensor([tok.encode("どうも仕方が")])
with torch.no_grad():
    z = model(idx)[0][0, -1]  # 最後の位置のロジット (V,)
top = torch.topk(z, 8).indices  # 温度 1 で上位 8 個のトークン
lines, rows = [], []
for tau in [0.5, 1.0, 2.0]:
    p = F.softmax(z / tau, dim=-1)  # 温度 tau の分布
    lines.append(f"温度 {tau}: " + ", ".join(
        f"{tok.decode([i])!r} {p[i]:.3f}" for i in top[:5].tolist()))
    rows.append([f"{p[i]:.4f}" for i in top.tolist()])
# [end: temperature]
(OUT / "temperature.txt").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
with open(OUT / "temperature.csv", "w", encoding="utf-8") as f:
    f.write("rank,t05,t10,t20\n")
    for r in range(8):
        f.write(f"{r + 1},{rows[0][r]},{rows[1][r]},{rows[2][r]}\n")
print("\n".join(lines))

# [snippet: sampling]
settings = [("貪欲（温度 0）", dict(temperature=0.0)),
            ("温度 0.5", dict(temperature=0.5)),
            ("温度 1.0", dict(temperature=1.0)),
            ("温度 1.5", dict(temperature=1.5)),
            ("温度 1.0, top-k 10", dict(temperature=1.0, top_k=10)),
            ("温度 1.0, top-p 0.9", dict(temperature=1.0, top_p=0.9))]
start = torch.tensor([tok.encode("\n　吾輩は")])
out_lines = []
for name, kw in settings:
    torch.manual_seed(3)  # どの設定でも同じ乱数の列を使う
    out = model.generate(start, max_new_tokens=30, **kw)
    s = tok.decode(out[0, start.size(1):].tolist()).replace("\n", "／")
    s = s.replace("\ufffd", "□")  # 文字にならないバイト列
    out_lines.append(f"[{name}] 吾輩は{s}")
# [end: sampling]
(OUT / "sampling.txt").write_text("\n".join(out_lines) + "\n",
                                  encoding="utf-8")
print("\n".join(out_lines))
