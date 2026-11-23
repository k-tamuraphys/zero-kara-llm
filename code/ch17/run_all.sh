#!/bin/bash
# 第 17 章の実験をまとめて回す（2 本ずつ並列、各 2 スレッド）
cd "$(dirname "$0")/.."
run() { python ch17/train_lora.py "$@" > ch17/out/lora_${4:-r$1_$2}.txt 2>&1; }
run 1 all 1e-3 & run 2 all 1e-3; wait
run 4 all 1e-3 & run 16 all 1e-3; wait
run 64 all 1e-3 & run 8 qkv 1e-3; wait
run 8 mlp 1e-3 & run 8 all 3e-3 r8_all_lr3e-3; wait
