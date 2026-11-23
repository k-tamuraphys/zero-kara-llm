"""基盤モデルを「計算アシスタント」に教師ありファインチューニング (SFT) する。

実行:  python scripts/sft_chat.py   （先に pretrain_base.py を実行）
出力:  checkpoints/sft.pt           CPU で 2〜3 分。
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.chat import (chat_accuracy, chat_reply, collate,
                          encode_chat_example, make_chat_example)
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import (load_checkpoint, lr_at, make_optimizer,
                           save_checkpoint)

CKPT_DIR = DATA_DIR.parent / "checkpoints"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
model = load_checkpoint(CKPT_DIR / "base.pt", device)
print(f"SFT 前の正答率: {chat_accuracy(model, tok, device=device):.1%}")
reply = chat_reply(model, tok, "12+35は？", device=device)
print("SFT 前の応答例:", repr(reply))

# 学習データ: 対話 3,000 件（評価用の問題とは乱数の種が異なる）
rng = random.Random(0)
examples = [encode_chat_example(tok, make_chat_example(rng)[0])
            for _ in range(3000)]
pad_id = tok.special_tokens["<|pad|>"]

max_steps, batch_size, max_lr = 600, 32, 3e-4
optimizer = make_optimizer(model, max_lr, weight_decay=0.0)
model.train()
for step in range(max_steps):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step, max_lr, warmup_steps=30, max_steps=max_steps)
    batch = rng.sample(examples, batch_size)
    x, y = collate(batch, pad_id)
    _, loss = model(x.to(device), y.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0 or step == max_steps - 1:
        print(f"step {step:4d} | loss {loss.item():.3f}")

save_checkpoint(model, CKPT_DIR / "sft.pt")
print(f"SFT 後の正答率: {chat_accuracy(model, tok, device=device):.1%}")
for q in ["12+35は？", "300から123を引くといくつですか？", "7足す8は？"]:
    print(q, "->", repr(chat_reply(model, tok, q, device=device)))
