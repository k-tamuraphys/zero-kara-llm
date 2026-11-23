"""BPE の素朴な実装。小さな例で学習・符号化・復号を確かめる。

実行:  python ch07/bpe_naive.py   （code/ ディレクトリで実行する）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ch07.capture import Capture
from llmbook.tokenizer import CharTokenizer, count_pairs, merge_pair

cap = Capture()


# [snippet: train]
def train_bpe(ids, vocab, n_merges):
    """ids に対してマージを n_merges 回行い、規則と語彙を育てる。"""
    merges = {}  # (a, b) -> 新しい ID。学習した順に並ぶ
    for k in range(n_merges):
        counts = count_pairs(ids)
        pair = max(counts, key=counts.get)  # 最も多い組
        new_id = len(vocab)  # 次の空き番号
        ids = merge_pair(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        print(f"{k + 1:2d}: {show(vocab[pair[0]])} + {show(vocab[pair[1]])}"
              f" -> {new_id} {show(vocab[new_id])}  {counts[pair]} 回")
    return merges, ids
# [end: train]


def show(token) -> str:
    """トークン 1 個を表示用の文字列にする。文字にならないバイトは 16 進で。"""
    if isinstance(token, str):
        return f"「{token}」"
    try:
        return f"「{token.decode('utf-8')}」"
    except UnicodeDecodeError:
        return f"<{token.hex(' ')}>"


# ---- (1) 文字から始める: 本文の手計算の検算 ----------------------------
text = "猫である。犬である。人であるが、鳥であった。"

# [snippet: toy-char]
ctok = CharTokenizer(text)  # 12 種類の文字に 0〜11 の番号を振る
ids = ctok.encode(text)  # 22 個の番号
vocab = dict(ctok.itos)  # 番号 -> 文字列。マージのたびに書き足す
merges, ids = train_bpe(ids, vocab, n_merges=3)
print(len(ids), "個:", " ".join(vocab[i] for i in ids))
# [end: toy-char]
cap.save("toy_char.txt")

# 手計算の表: 最初の組の数
counts = count_pairs(ctok.encode(text))
top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
for (a, b), n in top:
    print(f"({ctok.itos[a]}, {ctok.itos[b]}): {n}")
cap.save("toy_counts.txt")

# ---- (2) バイトから始める ----------------------------------------------
# [snippet: toy-byte]
ids = list(text.encode("utf-8"))  # 66 個のバイト（0〜255 の整数）
vocab = {i: bytes([i]) for i in range(256)}  # 最初の語彙は 256 種類のバイト
merges, ids = train_bpe(ids, vocab, n_merges=10)
print("66 個 ->", len(ids), "個")
# [end: toy-byte]
cap.save("toy_byte.txt")


# [snippet: encode-decode]
def encode(text, merges):
    """文字列をトークン ID の列にする。"""
    ids = list(text.encode("utf-8"))  # まずバイト列にする
    for pair, new_id in merges.items():  # 学習した順に規則を適用する
        ids = merge_pair(ids, pair, new_id)
    return ids


def decode(ids, vocab):
    """トークン ID の列を文字列に戻す。"""
    raw = b"".join(vocab[i] for i in ids)  # 各トークンのバイト列をつなぐ
    return raw.decode("utf-8", errors="replace")
# [end: encode-decode]


# [snippet: toy-encode]
for s in ["鳥である。", "魚である。"]:
    ids = encode(s, merges)
    pieces = " ".join(show(vocab[i]) for i in ids)
    print(f"{s} -> {ids}\n    {pieces}  元に戻るか: {decode(ids, vocab) == s}")
# [end: toy-encode]
cap.save("toy_encode.txt")
