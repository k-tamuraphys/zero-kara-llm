#!/bin/bash
# 1 つのスクリプトの出力を、本文に載せる単位に切り分ける（数値は変えない）。
set -e
O=ch04/out
# "---" の行で区切って name_1.txt, name_2.txt, ... にする
for name in value_demo torch_autograd; do
    awk -v p="$O/${name}_" 'BEGIN{n=1} /^---$/{n++; next} {print > (p n ".txt")}' \
        $O/$name.txt
done
grep -v "^step" $O/mlp_torch.txt > $O/mlp_torch_params.txt
grep "^step" $O/mlp_torch.txt > $O/mlp_torch_train.txt
awk 'NF==0{exit} {print}' $O/vanishing.txt > $O/vanishing_depth.txt
