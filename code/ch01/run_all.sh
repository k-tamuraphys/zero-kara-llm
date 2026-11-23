#!/bin/bash
# 第 1 章のスクリプトをすべて実行し、出力を out/ に保存する。
# 使い方: cd code && bash ch01/run_all.sh
# 各スクリプトの出力は "---" の行で区切られている。本文に載せやすいように、
# 区切りごとに out/<名前>_1.txt, out/<名前>_2.txt, ... にも分けて保存する。
set -eu
python ch01/dump_base.py > ch01/out/dump_base.txt   # 準備（PyTorch を使う）
for name in vectors numgrad prob sampling coin_mle entropy softmax; do
  # softmax.py は警告（標準エラー出力）も結果の一部なので、一緒に保存する
  python -u "ch01/$name.py" 2>&1 \
    | sed "s#$(pwd)/##" > "ch01/out/$name.txt"
  awk -v base="ch01/out/$name" '
    BEGIN { k = 1; file = base "_" k ".txt"; printf "" > file }
    /^---$/ { close(file); k++; file = base "_" k ".txt"; printf "" > file; next }
    { print > file }' "ch01/out/$name.txt"
done
