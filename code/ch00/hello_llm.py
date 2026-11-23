"""本書のゴールを先に体験する: 学習済みのミニGPT に続きを書かせる。"""

import torch

from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.manual_seed(0)
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")  # トークナイザ（第 7 章）
model = load_checkpoint(DATA_DIR.parent / "checkpoints" / "base.pt")
model.eval()

# [snippet: next-token]
prompt = "どうも仕方が"
ids = torch.tensor([tok.encode(prompt)])  # 文字列 → トークン番号の列
logits, _ = model(ids)  # 各位置での「次のトークン」のスコア
probs = torch.softmax(logits[0, -1], dim=-1)  # 最後の位置のスコア → 確率
top = torch.topk(probs, 5)
for p, i in zip(top.values, top.indices):
    print(f"{tok.decode([i])!r:12} {p.item():.3f}")
# [end: next-token]

print("---")
# [snippet: generate]
for prompt in ["\n　吾輩は", "\n12+35="]:
    ids = torch.tensor([tok.encode(prompt)])
    out = model.generate(ids, max_new_tokens=40, temperature=0.8, top_k=50)
    print(tok.decode(out[0].tolist()))  # トークン番号の列 → 文字列
# [end: generate]
