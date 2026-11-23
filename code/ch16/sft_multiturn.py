"""システムプロンプトと複数ターンを含む対話で SFT する。

実行:  python ch16/sft_multiturn.py    （続けて multiturn_test.py sft_multi）
出力:  checkpoints/ch16_sft_multi.pt   CPU で約 3 分。
"""

import random

import torch

from common import CKPT_DIR, load_base, run_sft, tok
from llmbook.chat import IGNORE, encode_chat_example, make_chat_example
from llmbook.train import save_checkpoint

SYSTEM = {"role": "system", "content": "あなたは計算アシスタントです。"}


# [snippet: multiturn-data]
def make_dialogue(rng: random.Random) -> tuple[list[int], list[int]]:
    """1〜3 ターンの対話。半分にはシステムプロンプトを付ける。"""
    messages = []
    for _ in range(rng.randint(1, 3)):
        messages += make_chat_example(rng)[0]  # 質問と応答を 1 組追加
    with_system = rng.random() < 0.5
    if with_system:
        messages = [SYSTEM] + messages
    ids, labels = encode_chat_example(tok, messages)
    if with_system:  # システムプロンプトは与えるもので、学習しない
        n_sys = len(tok.encode(f"<|system|>{SYSTEM['content']}<|end|>"))
        labels[:n_sys] = [IGNORE] * n_sys
    return ids, labels
# [end: multiturn-data]


torch.manual_seed(0)
rng = random.Random(0)
examples = [make_dialogue(rng) for _ in range(3000)]
model = run_sft(load_base(), examples)
save_checkpoint(model, CKPT_DIR / "ch16_sft_multi.pt")
