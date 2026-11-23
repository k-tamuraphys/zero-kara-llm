"""ブロードキャスト: 形の違うテンソルどうしの演算。"""

import torch

# [snippet: basic]
S = torch.tensor([[70.0, 80.0, 90.0],
                  [50.0, 60.0, 40.0]])  # 形 (2, 3)
print(S + 5)  # 数 1 個は、全要素に配られる
bonus = torch.tensor([0.0, 10.0, 20.0])  # 形 (3,)。科目ごとの加点
print(S + bonus)  # (2, 3) + (3,) → どの行にも同じ加点
mean = S.mean(dim=1, keepdim=True)  # 形 (2, 1)。人ごとの平均
print(S - mean)  # (2, 3) - (2, 1) → 各行から、その人の平均を引く
col = torch.tensor([[1.0], [2.0]])  # 形 (2, 1)
row = torch.tensor([[10.0, 20.0, 30.0]])  # 形 (1, 3)
print(col + row)  # (2, 1) + (1, 3) → 両方が引き伸ばされて (2, 3)
# [end: basic]

print("---")
# [snippet: fail]
m = S.mean(dim=1)  # keepdim を付け忘れた。形は (2,)
try:
    S - m  # (2, 3) と (2,) は、右端が 3 と 2 で合わない
except RuntimeError as e:
    print("RuntimeError:", e)
# [end: fail]
