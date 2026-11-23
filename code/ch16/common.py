"""第 16 章のスクリプトが共有する小さな道具。"""

import sys
from pathlib import Path

import random

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.corpus import DATA_DIR, load_corpus, train_val_split
from llmbook.tokenizer import BPETokenizer
from llmbook.chat import collate
from llmbook.train import (estimate_loss, load_checkpoint, lr_at,
                           make_optimizer)

CKPT_DIR = DATA_DIR.parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "out"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)  # CPU を他の作業と分け合う

tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
pad_id = tok.special_tokens["<|pad|>"]


def load_base():
    """事前学習済みの基盤モデル base.pt を読み込む。"""
    return load_checkpoint(CKPT_DIR / "base.pt", device)


def train_tokens() -> torch.Tensor:
    """事前学習に使ったトークン列（漱石 + 加減算ドリル）。"""
    arr = np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))


# [snippet: soseki-val]
_soseki_val = None


def soseki_val_loss(model) -> float:
    """漱石の検証用テキスト（学習に使っていない末尾 5%）での損失。"""
    global _soseki_val
    if _soseki_val is None:
        _, val_text = train_val_split(load_corpus())
        _soseki_val = torch.tensor(tok.encode(val_text))
    with torch.random.fork_rng():  # 学習側の乱数の流れを乱さない
        torch.manual_seed(1234)  # 毎回同じ窓で測り、比較をぶれにくくする
        return estimate_loss(model, _soseki_val, batch_size=32,
                             device=device)
# [end: soseki-val]


# [snippet: run-sft]
def run_sft(model, examples, max_steps=600, batch_size=32, max_lr=3e-4,
            seed=0):
    """sft_step_by_step.py と同じ学習ループを関数にしたもの（実験用）。"""
    rng = random.Random(seed)
    optimizer = make_optimizer(model, max_lr, weight_decay=0.0)
    model.train()
    for step in range(max_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, max_lr, warmup_steps=30,
                                max_steps=max_steps)
        x, y = collate(rng.sample(examples, min(batch_size, len(examples))),
                       pad_id)
        _, loss = model(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model
# [end: run-sft]
