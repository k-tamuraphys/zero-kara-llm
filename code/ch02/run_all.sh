#!/bin/bash
# 第 2 章のスクリプトをすべて実行し、出力を ch02/out/ に保存する。
# 使い方: cd code && bash ch02/run_all.sh
# 出力に "---" だけの行があれば、そこで区切って <名前>_1.txt, <名前>_2.txt, ... に分ける。
set -eu
cd "$(dirname "$0")/.."
mkdir -p ch02/out
for name in hello_tensor numpy_bridge shape_dtype random_seed model_tensors \
            indexing reshape reduce broadcast broadcast_bug matmul autograd \
            shape_errors softmax_ce speed; do
  if [ "$name" = speed ] && [ "${SKIP_SPEED:-0}" = 1 ]; then continue; fi
  start=$EPOCHREALTIME
  python "ch02/$name.py" > "ch02/out/$name.all.txt"
  if grep -qx -- '---' "ch02/out/$name.all.txt"; then
    awk -v base="ch02/out/$name" 'BEGIN{k=1} /^---$/{k++; next} {print > (base "_" k ".txt")}' \
      "ch02/out/$name.all.txt"
    rm "ch02/out/$name.all.txt"
  else
    mv "ch02/out/$name.all.txt" "ch02/out/$name.txt"
  fi
  awk -v n="$name" -v a="$start" -v b="$EPOCHREALTIME" 'BEGIN{printf "%-16s %6.1f 秒\n", n, b - a}'
done
