"""要素ごとの演算と、軸を指定した集約。"""

import torch

# [snippet: elementwise]
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(x + y, x * y)  # 足し算も掛け算も「要素ごと」
print(x ** 2, torch.exp(x))  # 関数も要素ごとに適用される
# [end: elementwise]

print("---")
# [snippet: reduce]
S = torch.tensor([[70.0, 80.0, 90.0],
                  [50.0, 60.0, 40.0]])  # 2 人 × 3 科目の点数。形 (2, 3)
print(S.sum())  # 全要素の合計
print(S.sum(dim=0))  # 軸 0 をつぶす → 科目ごとの合計。形 (3,)
print(S.sum(dim=1))  # 軸 1 をつぶす → 人ごとの合計。形 (2,)
print(S.mean(dim=1))  # 人ごとの平均
print(S.mean(dim=1, keepdim=True))  # つぶした軸を長さ 1 で残す。形 (2, 1)
print(S.argmax(dim=1))  # 人ごとに、最高点の科目の番号
# [end: reduce]
