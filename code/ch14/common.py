"""第 14 章の実験で共有する小道具: データ・小さなモデル・分散の初期化。"""

import os

import numpy as np
import torch
import torch.distributed as dist

from llmbook.corpus import DATA_DIR
from llmbook.model import GPT, GPTConfig


def load_bin(name: str) -> torch.Tensor:
    """data/{name}.bin（uint16 のトークン列）を int64 のテンソルで返す。"""
    arr = np.fromfile(DATA_DIR / f"{name}.bin", dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))


def tiny_gpt(seed: int = 0) -> GPT:
    """CPU で数秒で動く小さなミニGPT（2 層・64 次元・文脈長 64）。"""
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=4096, block_size=64, n_layer=2,
                    n_head=4, n_embd=64)
    return GPT(cfg)


def fixed_batch(data: torch.Tensor, step: int, B: int, T: int = 64):
    """ステップ番号から決まる「全体のバッチ」。どのプロセスでも同じものになる。"""
    g = torch.Generator().manual_seed(1000 + step)
    starts = torch.randint(len(data) - T - 1, (B,), generator=g)
    x = torch.stack([data[s:s + T] for s in starts])
    y = torch.stack([data[s + 1:s + T + 1] for s in starts])
    return x, y


# [snippet: setup]
def setup(rank: int, world_size: int, port: int = 29531) -> None:
    """1 台の PC の中で、world_size 個のプロセスを 1 つのグループにする。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 集合場所（ランク 0 の住所）
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)                 # 1 プロセス 1 コア
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
# [end: setup]


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])
