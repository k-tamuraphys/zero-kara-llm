"""SFT 済みの計算アシスタント sft.pt に、掛け算をたずねる。

実行:  python ch23/llm_limits.py  > ch23/out/llm_limits.txt   （約 20 秒）
"""

import torch

from common import CKPT_DIR, device, tok
from llmbook.chat import chat_accuracy, chat_reply
from llmbook.train import load_checkpoint

torch.manual_seed(0)
# [snippet: limits]
model = load_checkpoint(CKPT_DIR / "sft.pt", device)  # 第 16 章の到達点
for q in ["12+35は？", "37×48は？", "123掛ける45は？", "8×7は？"]:
    print(f"{q:12s} -> {chat_reply(model, tok, q, device=device)}")
for ops in ["+-", "*"]:
    acc = chat_accuracy(model, tok, ops=ops, device=device)  # 200 問
    print(f"演算 {ops:2s} の正答率: {acc:.1%}")
# [end: limits]
