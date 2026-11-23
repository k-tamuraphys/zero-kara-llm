#!/bin/bash
# 第 5 章のスクリプトをすべて実行し、出力を ch05/out/ に保存する。
# 使い方: cd code && bash ch05/run_all.sh
# （全部で 20〜25 分。CPU の混み具合で変わる。長いのは deep_train（1 回 3〜6 分。
#   乱数の種 1〜4 の 4 回は並べて実行）と deep_ablation（5 回を並べて約 10 分））
set -e
O=ch05/out
mkdir -p $O
for s in init_stats layernorm grad_flow deep_train optim optim2d optim_mlp \
         schedule clip regularize diagnose standard; do
    echo "== $s"
    python ch05/$s.py > $O/$s.txt
done
# 乱数の種を変えた 4 回を並べて実行し、種 0 と合わせて集計する（表 5.1）
for seed in 1 2 3 4; do
    python ch05/deep_train.py $seed > $O/deep_train_seed$seed.txt &
done
wait
python ch05/deep_summary.py > $O/deep_summary.txt
# 50 枚での確認の実験（正規化の位置・学習率・手法を変える）。同じく 5 回
for seed in 0 1 2 3 4; do
    python ch05/deep_ablation.py $seed > /dev/null &
done
wait
python ch05/deep_ablation.py summary > $O/deep_ablation.txt
# 「---」の行で区切られた出力を、掲載用に <名前>_1.txt、<名前>_2.txt … に分ける
for s in layernorm optim2d schedule clip regularize diagnose standard; do
    awk -v base="$O/${s}_" 'BEGIN{n=1} /^---$/{n++; next} {print > (base n ".txt")}' $O/$s.txt
done
