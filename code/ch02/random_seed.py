"""乱数と再現性、確率に従うサンプリング。"""

import torch

# [snippet: seed]
torch.manual_seed(0)  # 乱数の種を固定する
print(torch.randn(3))  # 標準正規分布から 3 個
print(torch.randn(3))  # 続けて呼ぶと、別の数が出る
torch.manual_seed(0)  # 同じ種に戻すと……
print(torch.randn(3))  # 最初と同じ数が出る
# [end: seed]

print("---")
# [snippet: sample]
p = torch.tensor([0.5, 0.3, 0.2])  # 番号 0, 1, 2 が出る確率
draws = torch.multinomial(p, num_samples=10000, replacement=True)
print(draws[:12])  # 最初の 12 回に出た番号
print(torch.bincount(draws) / 10000)  # 各番号が出た割合
# [end: sample]
