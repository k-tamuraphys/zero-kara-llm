"""第 17 章のスクリプトが共有する道具（第 16 章と同じ設定を使う）。"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.chat import (chat_accuracy, collate, encode_chat_example,
                          make_chat_example)
from llmbook.corpus import DATA_DIR, load_corpus, train_val_split
from llmbook.tokenizer import BPETokenizer
from llmbook.train import estimate_loss, load_checkpoint, lr_at

CKPT_DIR = DATA_DIR.parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "out"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)  # CPU を他の作業と分け合う

tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
pad_id = tok.special_tokens["<|pad|>"]
EVAL_N, EVAL_SEED = 1000, 777  # 比較は 1,000 問で（200 問では ±2〜3 点ぶれる）


def load_base():
    return load_checkpoint(CKPT_DIR / "base.pt", device)


def sft_examples(n=3000, seed=0, suffix="です。"):
    """第 16 章と同じ対話 3,000 件。suffix で応答の語尾を変えられる。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        messages, answer = make_chat_example(rng)
        messages[1]["content"] = f"{answer}{suffix}"
        out.append(encode_chat_example(tok, messages))
    return out


_soseki_val = None


def soseki_val_loss(model) -> float:
    """漱石の検証用テキストでの損失（第 16 章と同じ測り方）。"""
    global _soseki_val
    if _soseki_val is None:
        _, val_text = train_val_split(load_corpus())
        _soseki_val = torch.tensor(tok.encode(val_text))
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        return estimate_loss(model, _soseki_val, batch_size=32,
                             device=device)


def accuracy(model, n=EVAL_N) -> float:
    return chat_accuracy(model, tok, n_problems=n, seed=EVAL_SEED,
                         device=device)


# [snippet: train-loop]
def train(model, examples, max_steps=600, batch_size=32, max_lr=3e-4,
          seed=0, log=None):
    """第 16 章の SFT と同じループ。更新するのは requires_grad の値だけ。"""
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=max_lr, betas=(0.9, 0.95),
                                  weight_decay=0.0)
    model.train()
    for step in range(max_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, max_lr, warmup_steps=30,
                                max_steps=max_steps)
        x, y = collate(rng.sample(examples, batch_size), pad_id)
        _, loss = model(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if log is not None and (step % 25 == 0 or step == max_steps - 1):
            log.append((step, loss.item()))
    model.eval()
    return model
# [end: train-loop]
