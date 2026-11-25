"""図の手本 1 件で、正解にする位置ごとの損失を SFT の前後で比べる。

手本は本文の損失マスクの図と同じ「48と55を掛けるといくつですか？」。
sft.pt（ツールの SFT の前）と ch23_tool_sft.pt（後）で、緑の位置
（正解として当てるトークン）ごとに -log p(正解) を計算する。
実行:  python ch23/fig_token_loss.py > ch23/out/token_loss.txt   （数秒）
出力:  ch23/out/token_loss.csv
"""

import csv
import math

import torch
import torch.nn.functional as F

from common import CKPT_DIR, OUT_DIR, device, tok
from llmbook.chat import IGNORE, encode_chat_example
from llmbook.train import load_checkpoint

messages = [{"role": "user", "content": "48と55を掛けるといくつですか？"},
            {"role": "tool", "content": "48*55"},
            {"role": "result", "content": "2640"},
            {"role": "assistant", "content": "2640です。"}]
ids, labels = encode_chat_example(tok, messages, learn_role=True)
x = torch.tensor([ids], device=device)
y = torch.tensor(labels)
pos = [t for t in range(len(labels)) if labels[t] != IGNORE]  # 緑の位置

losses = {}
for name in ["sft", "ch23_tool_sft"]:
    model = load_checkpoint(CKPT_DIR / f"{name}.pt", device).eval()
    with torch.no_grad():
        logits, _ = model(x)                       # (1, T, V)
    logp = F.log_softmax(logits[0], dim=-1)        # (T, V)
    losses[name] = [-logp[t, y[t]].item() for t in pos]

print(f"当てずっぽうの損失 log 4096 = {math.log(4096):.2f}")
print("番号  正解のトークン        sft.pt  ch23_tool_sft.pt")
rows = []
for i, t in enumerate(pos):
    s = tok.decode([labels[t]])
    a, b = losses["sft"][i], losses["ch23_tool_sft"][i]
    print(f"{i + 1:3d}  {s:18s} {a:7.2f}  {b:7.3f}")
    rows.append({"i": i + 1, "sft": f"{a:.3f}", "tool": f"{b:.4f}"})
for name in ["sft", "ch23_tool_sft"]:
    v = losses[name]
    print(f"{name}: 緑の {len(v)} 位置の平均 {sum(v) / len(v):.2f}")
with open(OUT_DIR / "token_loss.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["i", "sft", "tool"])
    w.writeheader()
    w.writerows(rows)
