"""ブロードキャストが黙って起こす 2 つの典型的なバグ。"""

import torch

# [snippet: normalize]
C = torch.tensor([[8.0, 1.0, 1.0],
                  [2.0, 2.0, 0.0],
                  [1.0, 3.0, 4.0]])  # 回数の表。各行を「割合」に直したい
bad = C / C.sum(dim=1)  # (3, 3) / (3,)。エラーにならない
good = C / C.sum(dim=1, keepdim=True)  # (3, 3) / (3, 1)
print(bad.sum(dim=1))  # 各行の和が 1 にならない
print(good.sum(dim=1))  # 各行の和が 1 になる
# [end: normalize]

print("---")
# [snippet: outer]
pred = torch.tensor([[1.0], [2.0], [3.0]])  # 予測。形 (3, 1)
target = torch.tensor([1.0, 2.0, 3.0])  # 正解。形 (3,)
diff = pred - target  # 予測と正解は同じ値なので、全部 0 のはず
print(diff.shape)  # (3, 3)。3 個の差のはずが、9 個ある
print((diff ** 2).mean())  # 誤差の平均が 0 にならない
fixed = pred.squeeze(1) - target  # 形を (3,) にそろえてから引く
print((fixed ** 2).mean())
# [end: outer]
