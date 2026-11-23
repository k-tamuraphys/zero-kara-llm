"""ベクトルと行列を NumPy で確かめ、最後に LLM の「最後の一歩」を再現する。"""

import numpy as np

# [snippet: basics]
a = np.array([3.0, 4.0])  # ベクトル a
b = np.array([4.0, 3.0])  # ベクトル b
print("a・b   =", a @ b)  # 内積: 3*4 + 4*3
print("||a||  =", np.linalg.norm(a))  # ノルム: sqrt(3^2 + 4^2)
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print("cos    =", cos)  # コサイン類似度

A = np.array([[2.0, 1.0],
              [0.0, 1.0]])  # 2 行 2 列の行列
B = np.array([[0.0, -1.0],
              [1.0, 0.0]])  # 90 度の回転
x = np.array([1.0, 1.0])
print("A x     =", A @ x)  # 行列とベクトルの積
print("B (A x) =", B @ (A @ x))  # A で移してから B で移す
print("(B A) x =", (B @ A) @ x)  # 先に行列どうしを掛けても同じ
print("(A B) x =", (A @ B) @ x)  # 順番を変えると結果が変わる
M, N = np.ones((3, 2)), np.ones((2, 4))  # 形が (3, 2) と (2, 4) の行列
print("M N の形:", (M @ N).shape)
# [end: basics]

print("---")
# [snippet: similarity]
data = np.load("ch01/out/last_step.npz")
W = data["W"]  # 形は (4096, 192)。1 行が 1 トークンのベクトル
tokens = data["tokens"].tolist()  # 4096 個のトークンの文字列
index = {t: i for i, t in enumerate(tokens)}  # トークン → 番号


def cosine(u, v):
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))


pairs = [("7", "8"), ("先生", "奥さん"), ("東京", "京都"),
         ("7", "先生"), ("8", "東京"), ("奥さん", "京都")]
for s, t in pairs:
    c = cosine(W[index[s]], W[index[t]])
    print(f"cos({s}, {t}) = {c:+.2f}")
# [end: similarity]

# 図（ヒートマップ）用に、6 トークンどうしの類似度を CSV に保存する
words = ["7", "8", "先生", "奥さん", "東京", "京都"]
with open("ch01/out/similarity.csv", "w", encoding="utf-8") as f:
    f.write("i,j,cos\n")
    for i, s in enumerate(words):
        for j, t in enumerate(words):
            f.write(f"{i},{j},{cosine(W[index[s]], W[index[t]]):.2f}\n")

print("---")
# [snippet: last-step]
h = data["h"]  # 「どうも仕方が」を読み終えたモデルが作ったベクトル
z = W @ h  # スコア。z[k] は「W の k 行目」と h の内積
print("W:", W.shape, " h:", h.shape, " z:", z.shape)
k = np.argmax(z)  # スコアが最大になる番号
print("argmax =", k, repr(tokens[k]), " max =", round(float(z.max()), 2))
for i in np.argsort(-z)[:5]:  # スコアの大きい順に 5 個
    print(f"{tokens[i]!r:10} {z[i]:6.2f}")
# [end: last-step]
