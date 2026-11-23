#!/bin/bash
# 第 6 章のスクリプトをすべて実行し、出力を ch06/out/ に保存する。
# 使い方: cd code && bash ch06/run_all.sh
# 出力に "---" だけの行があれば、そこで区切って <名前>_1.txt, <名前>_2.txt, ... に分ける。
set -eu
cd "$(dirname "$0")/.."
mkdir -p ch06/out
for name in toy_bigram bigram_count bigram_nn ngram sampling preview_gpt; do
  start=$EPOCHREALTIME
  python "ch06/$name.py" > "ch06/out/$name.all.txt"
  if grep -qx -- '---' "ch06/out/$name.all.txt"; then
    awk -v base="ch06/out/$name" 'BEGIN{k=1} /^---$/{k++; next} {print > (base "_" k ".txt")}' \
      "ch06/out/$name.all.txt"
    rm "ch06/out/$name.all.txt"
  else
    mv "ch06/out/$name.all.txt" "ch06/out/$name.txt"
  fi
  awk -v n="$name" -v a="$start" -v b="$EPOCHREALTIME" 'BEGIN{printf "%-14s %6.1f 秒\n", n, b - a}'
done
