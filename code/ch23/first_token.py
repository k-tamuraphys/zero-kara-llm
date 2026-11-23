"""質問を読んだ直後の「最初の 1 トークン」の確率を調べる。

<|user|>質問<|end|> の次のトークンが <|tool|>（電卓を呼ぶ）になる確率と
<|assistant|>（自分で答える）になる確率を、SFT の前後で比べる。
実行:  python ch23/first_token.py  > ch23/out/first_token.txt  （数秒）
"""

import torch
import torch.nn.functional as F

from common import CKPT_DIR, device, tok
from llmbook.train import load_checkpoint

TOOL = tok.special_tokens["<|tool|>"]
ASSISTANT = tok.special_tokens["<|assistant|>"]
questions = ["37×48は？", "123掛ける45は？",
             "25×4に7を足すといくつですか？", "12+35は？", "63-4は？"]

for name in ["sft", "ch23_tool_sft"]:
    model = load_checkpoint(CKPT_DIR / f"{name}.pt", device).eval()
    print(f"{name}.pt: 最初の 1 トークンの確率")
    # [snippet: first-token]
    for q in questions:
        idx = torch.tensor([tok.encode(f"<|user|>{q}<|end|>")], device=device)
        with torch.no_grad():
            logits, _ = model(idx)             # (1, T, V)
        p = F.softmax(logits[0, -1], dim=-1)   # 次のトークンの確率 (V,)
        print(f"  {q:22s} <|tool|> {p[TOOL]:.3f}"
              f"  <|assistant|> {p[ASSISTANT]:.3f}")
    # [end: first-token]
