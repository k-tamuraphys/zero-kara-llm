"""第 8 章: 学習した埋め込みを観察する（近いトークン、2 次元への射影）。

実行: cd code && python ch08/embedding_neighbors.py   （先に mlp_lm.py を実行）
出力: ch08/out/nb_*.txt, ch08/out/pca_*.csv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from capture import OUT, Capture
from mlp_lm import CKPT, V, tok, train

torch.manual_seed(0)
cap = Capture()

# [snippet: load]
E = torch.load(CKPT)["model"]["emb.weight"]   # (V, d) 学習した埋め込み表
En = F.normalize(E, dim=1)                    # 各行を長さ 1 にそろえる
count = torch.bincount(train, minlength=V)    # 訓練データでの出現回数
# [end: load]
# 観察の対象から外すトークン: まれなもの、文字の途中で切れているもの
skip = (count < 50) | torch.tensor(
    ["\ufffd" in tok.token_str(i) for i in range(V)])
print("表の形", tuple(E.shape),
      f" 観察するトークン {(~skip).sum().item():,} 個")
cap.save("nb_load.txt")


def tid(word):
    ids = tok.encode(word)
    assert len(ids) == 1, f"{word} は 1 トークンではない"
    return ids[0]


# [snippet: neighbors]
def neighbors(table, word, k=5):
    """word に対応する行と、コサイン類似度が高い行を k 個返す。"""
    sim = table @ table[tid(word)]            # (V,) 全トークンとの類似度
    sim[skip] = -1                            # まれなトークンなどは除く
    val, idx = sim.topk(k + 1)                # 先頭は自分自身（類似度 1）
    return [(tok.token_str(int(i)).replace("\n", "\\n"), float(v))
            for v, i in zip(val[1:], idx[1:])]
# [end: neighbors]


words = ["7", "三", "先生", "私", "です。", "は", "春", "東京"]
for w in words:
    body = " ".join(f"{s}({v:.2f})" for s, v in neighbors(En, w))
    print(f"「{w}」→ {body}")
cap.save("nb_table.txt")

# [snippet: pairs]
def cos(a, b):
    return float(En[tid(a)] @ En[tid(b)])


for a, b in [("7", "8"), ("先生", "奥さん"), ("東京", "京都"), ("7", "先生")]:
    print(f"「{a}」と「{b}」: {cos(a, b):+.2f}")
# [end: pairs]
cap.save("nb_pairs.txt")

# 比較: 学習前（乱数のまま）の表では、いちばん近いトークンでもどの程度か
E0 = F.normalize(nn.Embedding(V, E.shape[1]).weight.detach(), dim=1)
top0 = [neighbors(E0, w, k=1)[0][1] for w in words]
top1 = [neighbors(En, w, k=1)[0][1] for w in words]
print(f"いちばん近いトークンとの類似度（{len(words)} 個の平均）")
print(f"  学習前の表 {sum(top0) / len(top0):.2f}"
      f"   学習後の表 {sum(top1) / len(top1):.2f}")
cap.save("nb_random.txt")

# --- 2 次元への射影（主成分分析）---
groups = {
    "digit": list("0123456789"),
    "kanji": ["一", "二", "三", "四", "五", "六", "七", "八", "十"],
    "person": ["先生", "奥さん", "私", "僕", "彼", "彼女", "自分", "君",
               "Ｋ", "代助", "三四郎"],
    "ending": ["です。", "ます。", "である。", "いた。", "した。", "ない。",
               "ある。", "った。"],
    "particle": ["は", "が", "を", "に", "と", "も"],
}
ids = [tid(w) for ws in groups.values() for w in ws]
# [snippet: pca]
X = E[ids]                                    # (m, d) 選んだトークンの行
X = X - X.mean(dim=0)                         # 平均を原点に移す
U, S, Vt = torch.linalg.svd(X, full_matrices=False)
xy = X @ Vt[:2].T                             # (m, 2) ばらつきの大きい 2 方向
# [end: pca]
ratio = (S[:2] ** 2).sum() / (S ** 2).sum()
print(f"選んだ {len(ids)} 個のトークンを 2 次元に射影。"
      f"ばらつきの {ratio:.0%} を保持")
cap.save("nb_pca.txt")
k = 0
for name, ws in groups.items():
    with open(OUT / f"pca_{name}.csv", "w", encoding="utf-8") as f:
        f.write("x,y,label\n")
        for w in ws:
            f.write(f"{xy[k, 0]:.3f},{xy[k, 1]:.3f},{w}\n")
            k += 1
