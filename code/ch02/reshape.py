"""形を変える: reshape / view / transpose / permute など。"""

import torch

# [snippet: reshape]
x = torch.arange(12)  # 0, 1, ..., 11 の 12 個
print(x.reshape(3, 4))  # 4 個ずつ区切って 3 行に
print(x.reshape(2, 6).shape, x.reshape(2, 2, 3).shape)  # ほかの区切り方
print(x.reshape(4, -1).shape)  # -1 は「残りから計算して」の意味
# [end: reshape]

print("---")
# [snippet: transpose]
A = torch.arange(6).reshape(2, 3)
print(A)
print(A.T)  # 転置: 行と列を入れ替える
print(A.reshape(3, 2))  # 形は同じ (3, 2) でも、中身の並びが違う
# [end: transpose]

print("---")
# [snippet: view]
B = A.T  # 転置は、メモリを並べ替えずに「読む順番」だけを変える
print(B.is_contiguous())  # メモリの順と読む順が一致しているか
try:
    B.view(6)  # view は並べ替えをしないので、ここで失敗する
except RuntimeError as e:
    print("RuntimeError:", e)
print(B.reshape(6))  # reshape は、必要ならコピーして並べ直す
# [end: view]

print("---")
# [snippet: permute]
X = torch.zeros(2, 3, 4)  # 形 (2, 3, 4)
print(X.transpose(1, 2).shape)  # 軸 1 と軸 2 を入れ替える
print(X.permute(2, 0, 1).shape)  # 軸を「2, 0, 1」の順に並べ替える
print(X.unsqueeze(0).shape, X[None].shape)  # 先頭に長さ 1 の軸を足す
print(X[None].squeeze(0).shape)  # 長さ 1 の軸を取り除く
print(X.flatten().shape)  # 1 列に伸ばす
print(torch.cat([X, X], dim=0).shape)  # 軸 0 の方向につなげる
# [end: permute]
