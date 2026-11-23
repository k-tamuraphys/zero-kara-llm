"""実務での書き方: peft ライブラリ（0.22 系）で同じ LoRA を作る。

実行:  python ch17/peft_example.py   （600 ステップ学習して自作版と比べる）
出力:  ch17/out/peft_example.txt
"""

import torch

from common import accuracy, load_base, sft_examples, train

# [snippet: peft]
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0,
    target_modules=["qkv", "proj", "fc"],  # 名前がこれで終わる層に付ける
)
model = get_peft_model(load_base(), config)  # 元の重みは自動で凍結
model.print_trainable_parameters()
# [end: peft]
qkv = model.base_model.model.blocks[0].attn.qkv  # 差し替え後の qkv 層
print(type(qkv).__name__, "| W0:", tuple(qkv.base_layer.weight.shape),
      "| A:", tuple(qkv.lora_A["default"].weight.shape),
      "| B:", tuple(qkv.lora_B["default"].weight.shape))

torch.manual_seed(0)
train(model, sft_examples(), max_steps=600, max_lr=1e-3)
print(f"対話正答率（1,000 問）: {accuracy(model):.1%}")
# [snippet: peft-merge]
merged = model.merge_and_unload()  # W + (alpha/r) B A を書き込み、元の型へ
print(type(merged).__name__, sum(p.numel() for p in merged.parameters()))
# model.save_pretrained("my_adapter")  # 実務ではアダプタだけを保存する
# [end: peft-merge]
