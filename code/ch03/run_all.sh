#!/bin/bash
# 第 3 章のスクリプトをすべて実行し、本文に載せる出力を out/ に保存する。
# 使い方: cd code && bash ch03/run_all.sh   （全部で 2 分ほど）
set -e
O=ch03/out
mkdir -p $O
cut_lines() {  # cut_lines 元ファイル 開始行 終了行 出力名
  sed -n "$2,$3p" "$O/$1" > "$O/$4"
}

[ -f $O/char_token_big.csv ] || python ch03/make_data.py > $O/make_data.txt          # 約 90 秒（トークン化）
python ch03/hand_example.py > $O/hand_example.txt
cut_lines hand_example.txt 1 6 hand_loss.txt
cut_lines hand_example.txt 10 17 hand_gd.txt
python ch03/linreg_numpy.py > $O/linreg_numpy.txt
cut_lines linreg_numpy.txt 1 3 linreg_check.txt
cut_lines linreg_numpy.txt 4 11 linreg_run.txt
cut_lines linreg_numpy.txt 12 14 linreg_closed.txt
cut_lines linreg_numpy.txt 15 16 linreg_test.txt
cut_lines linreg_numpy.txt 17 21 linreg_lr.txt
python ch03/linreg_torch.py > $O/linreg_torch.txt
cut_lines linreg_torch.txt 1 2 linreg_torch_grad.txt
cut_lines linreg_torch.txt 3 6 linreg_torch_loop.txt
python ch03/learn_addition.py > $O/learn_addition.txt
python ch03/gauss_mle.py > $O/gauss_mle.txt
cut_lines gauss_mle.txt 1 6 mle_table.txt
cut_lines gauss_mle.txt 8 9 mle_resid.txt
python ch03/logistic_regression.py > $O/logistic_regression.txt
cut_lines logistic_regression.txt 1 6 logistic_train.txt
cut_lines logistic_regression.txt 7 13 logistic_predict.txt
python ch03/softmax_regression.py > $O/softmax_regression.txt
cut_lines softmax_regression.txt 1 4 softmax_check.txt
cut_lines softmax_regression.txt 5 12 softmax_train.txt
cut_lines softmax_regression.txt 13 14 softmax_predict.txt
python ch03/sgd_minibatch.py > $O/sgd_minibatch.txt
cut_lines sgd_minibatch.txt 1 6 sgd_noise.txt
cut_lines sgd_minibatch.txt 8 16 sgd_compare.txt
python ch03/overfit_poly.py > $O/overfit_poly.txt
cut_lines overfit_poly.txt 1 11 poly_degree.txt
sed -n '1p;3p;5p;7p;9p;11p' $O/overfit_poly.txt > $O/poly_degree_odd.txt  # 奇数の次数だけ（本文掲載用）
cut_lines overfit_poly.txt 13 19 poly_reg.txt
cut_lines overfit_poly.txt 21 26 poly_more.txt
cut_lines overfit_poly.txt 28 29 poly_test.txt
python ch03/linear_limit.py > $O/linear_limit.txt
