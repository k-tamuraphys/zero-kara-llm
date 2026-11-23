"""最初の動作確認: テンソルを作って計算し、最後に版を表示する。"""

import numpy as np

# [snippet: first]
import torch

x = torch.tensor([1.0, 2.0, 3.0])  # Python のリストからテンソルを作る
y = torch.tensor([10.0, 20.0, 30.0])
print(x + y)  # 要素ごとの足し算
print(2 * x)  # スカラー倍
print(x @ y)  # 内積 1*10 + 2*20 + 3*30
# [end: first]

print("---")
print("PyTorch", torch.__version__, " NumPy", np.__version__)
