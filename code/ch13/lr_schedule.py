"""基盤モデルの学習率スケジュール（lr_at）を表にする。図を描くためのもの。

実行: python ch13/lr_schedule.py  （1 秒）
出力: ch13/out/lr_schedule.csv
"""

from llmbook.train import lr_at

with open("ch13/out/lr_schedule.csv", "w") as f:
    f.write("step,lr\n")
    for step in range(0, 4001, 20):
        f.write(f"{step},{lr_at(step, 1e-3, 200, 4000):.6g}\n")
