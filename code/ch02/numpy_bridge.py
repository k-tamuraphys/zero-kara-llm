"""NumPy の配列と PyTorch のテンソルを行き来する。"""

import numpy as np
import torch

# [snippet: bridge]
a = np.array([1.0, 2.0, 3.0])  # NumPy の配列
t = torch.from_numpy(a)  # NumPy → PyTorch（コピーせず、メモリを共有する）
b = t.numpy()  # PyTorch → NumPy（これも共有）
print(a + a, t + t)  # 書き方はほとんど同じ
a[0] = 99.0  # 元の配列を書き換えると……
print(t)  # テンソルの側も変わる
c = torch.tensor(a)  # torch.tensor はコピーを作る
a[0] = 1.0
print(t, c)  # 共有している t は戻り、コピーの c は 99 のまま
# [end: bridge]
