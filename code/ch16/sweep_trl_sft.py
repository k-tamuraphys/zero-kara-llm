"""TRL の SFTTrainer を、この計算機（CPU のみ）で実際に動かして確かめる。

本文の 6 行（Qwen/Qwen3-0.6B を Capybara で SFT）と同じ書き方で、次の 3 点
だけを小さくする。
  - モデル: Qwen3 と同じ形・同じトークナイザで層を極端に小さくした、
    重みが乱数の検査用モデル trl-internal-testing/tiny-Qwen3ForCausalLM
    （Qwen3-0.6B は、AdamW で学習するとメモリ 8 GB に収まらない）
  - データ: Capybara の先頭 16 件
  - 学習: 2 ステップだけ（バッチ 4）
動くこと（テンプレートの適用・損失マスク・学習ループ）を確かめるのが目的で、
重みが乱数なので、賢いモデルはできない。
必要なもの: trl（2026-11-23 に trl 1.18.2 で確認）。初回はデータと
モデルのダウンロード（数十 MB）がある。
実行:  python ch16/sweep_trl_sft.py   出力: ch16/out/sweep_trl_sft.txt
"""

import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

OUT = Path(__file__).resolve().parent / "out" / "sweep_trl_sft.txt"
torch.manual_seed(0)
t0 = time.time()
lines = []

dataset = load_dataset("trl-lib/Capybara", split="train").select(range(16))
args = SFTConfig(output_dir="/tmp/ch16_trl_sft", assistant_only_loss=True,
                 max_steps=2, per_device_train_batch_size=4, max_length=256,
                 logging_steps=1, save_strategy="no", report_to="none",
                 use_cpu=True, seed=0)
trainer = SFTTrainer("trl-internal-testing/tiny-Qwen3ForCausalLM",
                     train_dataset=dataset, args=args)
tok = trainer.processing_class

# 1 件目の手本: 何トークンのうち、何トークンが損失の対象（正解が -100 以外）か
ex = trainer.train_dataset[0]
n_all = len(ex["input_ids"])
n_learn = sum(label != -100 for label in ex["labels"])
lines.append(f"手本 1 件目: {n_all} トークンのうち、"
             f"損失を計算するのは {n_learn} トークン（アシスタントの部分）")

# 本章の対話を、同じ仕組み（学習用テンプレートと assistant_only_loss）で
# 処理し、どのトークンが損失の対象になるかを表示する
messages = [{"role": "user", "content": "12+35は？"},
            {"role": "assistant", "content": "47です。"}]
enc = tok.apply_chat_template(messages, chat_template=trainer.chat_template,
                              tokenize=True, return_dict=True,
                              return_assistant_tokens_mask=True)
pieces = []
for i, m in zip(enc["input_ids"], enc["assistant_masks"]):
    piece = tok.decode([i]).replace("\n", "⏎")
    pieces.append(f"[{piece}]" if m else piece)
lines.append("「12+35は？」の対話（[ ] が損失の対象）:")
lines.append("".join(pieces))

result = trainer.train()
for log in trainer.state.log_history:
    if "loss" in log:
        lines.append(f"step {log['step']} | 損失 {log['loss']:.3f}")
vocab = len(tok)
lines.append(f"（語彙 {vocab:,} 個。でたらめな予測の損失 "
             f"log {vocab:,} ≈ {math.log(vocab):.3f}）")
lines.append(f"所要時間 {time.time() - t0:.0f} 秒"
             f"（うち学習 {result.metrics['train_runtime']:.0f} 秒）")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
