#!/bin/bash
# 第 4 章のスクリプトをすべて実行し、出力を ch04/out/ に保存する。
# 使い方: cd code && bash ch04/run_all.sh   （全部で 5〜15 分。CPU の混み具合で変わる）
set -e
O=ch04/out
mkdir -p $O
for s in linear_limit stack_linear xor_by_hand activations relu_approx \
         value_demo check_vs_torch backprop_layer torch_autograd \
         cost_compare mlp_scratch scratch_vs_torch mlp_torch vanishing; do
    echo "== $s"
    python ch04/$s.py > $O/$s.txt
done
bash ch04/split_out.sh
