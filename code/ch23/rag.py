"""検索拡張生成（RAG）の「検索」部分を自作する。

漱石コーパスの段落を文字 2-gram の TF-IDF ベクトル（素朴な埋め込み）にし、
質問とのコサイン類似度で上位の段落を選んで、文脈に差し込む。
実行:  python ch23/rag.py  > ch23/out/rag.txt          （約 30 秒）
"""

import math
from collections import Counter

import warnings

import torch

from common import tok  # noqa: F401  （スレッド数の設定のため）
from llmbook.corpus import load_corpus

warnings.filterwarnings("ignore", category=UserWarning)  # 疎行列の beta 警告

# [snippet: rag-index]
docs = [p.strip("　") for p in load_corpus().split("\n") if len(p) >= 20]


# [snippet: rag-embed]
def bigrams(text: str) -> Counter:
    return Counter(text[i:i + 2] for i in range(len(text) - 1))


counts = [bigrams(d) for d in docs]
df = Counter(g for c in counts for g in c)       # その 2-gram を含む段落数
vocab = {g: j for j, g in enumerate(df)}
idf = {g: math.log(len(docs) / n) for g, n in df.items()}  # 珍しいほど大


def embed(c: Counter) -> dict[int, float]:
    """TF-IDF ベクトル（0 でない成分だけ）を長さ 1 に正規化して返す。"""
    v = {vocab[g]: n * idf[g] for g, n in c.items() if g in vocab}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {j: x / norm for j, x in v.items()}
# [end: rag-embed]


rows, cols, vals = [], [], []
for i, c in enumerate(counts):
    for j, x in embed(c).items():
        rows.append(i), cols.append(j), vals.append(x)
E = torch.sparse_coo_tensor([rows, cols], vals,
                            (len(docs), len(vocab))).to_sparse_csr()
# [end: rag-index]


# [snippet: rag-search]
# [snippet: rag-search-only]
def search(query: str, k: int = 3) -> list[tuple[float, str]]:
    """質問に近い段落を k 個、類似度つきで返す。"""
    q = torch.zeros(len(vocab))
    for j, x in embed(bigrams(query)).items():
        q[j] = x
    scores = E @ q                       # (段落数,) = 各段落とのコサイン
    top = torch.topk(scores, k)
    return [(s.item(), docs[i]) for s, i in zip(top.values, top.indices)]
# [end: rag-search-only]


def rag_prompt(query: str, k: int = 2) -> str:
    """見つけた段落を「参考資料」として質問の前に差し込む。"""
    refs = "\n".join(f"[{n + 1}] {d[:60]}…"
                     for n, (_, d) in enumerate(search(query, k)))
    return (f"<|system|>次の資料だけを根拠に答えてください。\n{refs}<|end|>"
            f"<|user|>{query}<|end|><|assistant|>")
# [end: rag-search]


if __name__ == "__main__":
    print(f"段落数 {len(docs):,}、"
          f"2-gram の種類（ベクトルの次元）{len(vocab):,}")
    for query in ["どこで生れたか見当がつかない",
                  "親譲りの無鉄砲で損ばかりしている",
                  "山路を登りながら考えた", "先生と私は鎌倉の海で出会った",
                  "猫の名前は何ですか"]:
        print(f"■ 質問: {query}")
        for s, d in search(query):
            print(f"  {s:.3f}  {d[:34]}…")
    print("■ 文脈に差し込んだ形")
    print(rag_prompt("猫の名前は何ですか"))
