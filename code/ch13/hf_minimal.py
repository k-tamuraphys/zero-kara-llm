"""transformers で base.pt と同じ形の GPT-2 を作り、損失と勾配を求める。

実行: python ch13/hf_minimal.py  （約 10 秒。ダウンロードは不要）
出力: ch13/out/hf_minimal.txt
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ch13.common import load_bin
from llmbook.train import get_batch

torch.manual_seed(0)
# [snippet: hf]
config = GPT2Config(vocab_size=4096, n_positions=128, n_embd=192,
                    n_layer=4, n_head=4,  # base.pt と同じ大きさ
                    bos_token_id=4088, eos_token_id=4088)
model = GPT2LMHeadModel(config)
print(f"パラメータ数: {model.num_parameters():,}")
x, _ = get_batch(load_bin("train"), 4, 128)
out = model(input_ids=x, labels=x)  # labels は内部で 1 つずらされる
out.loss.backward()
print(f"学習前の損失: {out.loss.item():.3f}")
# [end: hf]
