"""図「温度と次トークンの確率」の横軸に書く、上位 8 個のトークンの名前。

sampling_demo.py と同じく、基盤モデルに「どうも仕方が」を与え、
温度 1 で上位 8 個のトークンを順に書き出す（確率は temperature.csv）。
実行: python ch10/fig_temp_labels.py
出力: ch10/out/fig_temp_labels.csv
"""

from pathlib import Path

import torch

from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

OUT = Path(__file__).parent / "out"
torch.set_num_threads(2)
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt").eval()
idx = torch.tensor([tok.encode("どうも仕方が")])
with torch.no_grad():
    z = model(idx)[0][0, -1]  # 最後の位置のロジット (V,)
top = torch.topk(z, 8).indices.tolist()  # 温度 1 で上位 8 個
with open(OUT / "fig_temp_labels.csv", "w", encoding="utf-8") as f:
    f.write("rank,token\n")
    for r, i in enumerate(top, start=1):
        s = tok.decode([i]).replace("\n", "＼n").replace("�", "□")
        f.write(f"{r},{s}\n")
print((OUT / "fig_temp_labels.csv").read_text(encoding="utf-8"))
