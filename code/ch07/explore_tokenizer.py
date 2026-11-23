"""本書のトークナイザ（data/bpe4096.json）を観察する。

実行:  python ch07/explore_tokenizer.py   （code/ ディレクトリで実行する）
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ch07.capture import OUT, Capture
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer

cap = Capture()


def show(token: bytes) -> str:
    """トークン 1 個を表示用の文字列にする。文字にならないバイトは 16 進で。"""
    try:
        return token.decode("utf-8").replace("\n", "\\n")
    except UnicodeDecodeError:
        return f"<{token.hex(' ')}>"


tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
print("語彙の大きさ:", tok.vocab_size)
print("マージ規則の数:", len(tok.merges))
cap.save("vocab_size.txt")
# [snippet: load]
for (a, b), new_id in list(tok.merges.items())[:8]:  # 学習した順に 8 個
    print(f"{show(tok.vocab[a]):9s} + {show(tok.vocab[b]):5s} -> "
          f"{new_id} {show(tok.vocab[new_id])}")
# [end: load]
cap.save("first_merges.txt")

# いつ生まれたか（表の数値）
where = {v: i for i, v in tok.vocab.items()}
for s in ["の", "。", "た。", "って", "ない", "である。", "ない。", "先生",
          "である", "吾輩", "猫", "吾輩は", "んでいた。"]:
    i = where.get(s.encode("utf-8"))
    print(f"{s}: ID {i}, マージ {i - 255} 番目" if i else f"{s}: なし")
made_from = {new_id: pair for pair, new_id in tok.merges.items()}
for s in ["である。", "である"]:  # 何と何から作られたか
    a, b = made_from[where[s.encode("utf-8")]]
    print(f"{s} = {show(tok.vocab[a])}（マージ {a - 255} 番目）"
          f" + {show(tok.vocab[b])}（マージ {b - 255} 番目）")
cap.save("born.txt")


def pieces(text: str) -> str:
    """text をトークンに切り、切れ目に / を入れて表示する。"""
    ids = tok.encode(text)
    cut = "/".join(show(tok.vocab[i]) for i in ids)
    return f"{cut}  （{len(text)} 字 -> {len(ids)} トークン）"


# [snippet: pieces]
print(pieces("吾輩は猫である。名前はまだ無い。"))
print(pieces("どうも仕方がない。"))
print(pieces("私はその人を常に先生と呼んでいた。"))
# [end: pieces]
cap.save("pieces.txt")

# [snippet: drill-unknown]
print(tok.encode("12+35=47\n"))  # 計算ドリルの 1 行（最後は改行）
for s in ["餃子", "薔薇", "LLM", "\U0001F600"]:  # 最後は笑顔の絵文字
    print(pieces(s))
# [end: drill-unknown]
cap.save("drill_unknown.txt")
for s in ["12+35=47\n", "餃子", "薔薇", "LLM", "\U0001F600"]:
    assert tok.decode(tok.encode(s)) == s  # どれも元に戻る

# [snippet: special]
text = "<|user|>12+35は？<|end|><|assistant|>47です。<|end|>"
ids = tok.encode(text)
print(ids)
print(" ".join(tok.token_str(i) for i in ids))
# [end: special]
cap.save("special.txt")

# [snippet: prefix]
print(pieces("吾輩は猫である。"))
print(pieces("吾輩は猫で"))
# [end: prefix]
cap.save("prefix.txt")

# ---- 語彙を途中で打ち切ったら、同じ文はどう切れるか（図の数値）----------
from llmbook.corpus import load_corpus
from llmbook.tokenizer import merge_pair

sent = "吾輩は猫である。名前はまだ無い。"
rules = list(tok.merges.items())
for size in [256, 512, 1024, 2048, 4088]:
    ids = list(sent.encode("utf-8"))
    for pair, new_id in rules[:size - 256]:
        ids = merge_pair(ids, pair, new_id)
    cut = "/".join(show(tok.vocab[i]) for i in ids)
    shown = cut if size > 256 else "（48 バイト）"
    print(f"{size:5d}: {len(ids):2d} 個  {shown}")
cap.save("levels.txt")

# ---- 文字の出現回数の偏り（本文の数値）-----------------------------------
text = load_corpus()
cfreq = Counter(text)
once = sum(1 for c in cfreq.values() if c == 1)
few = sum(1 for c in cfreq.values() if c <= 10)
top100 = sum(c for _, c in cfreq.most_common(100)) / len(text)
print(f"文字の種類 {len(cfreq)}、1 回しか現れない字 {once}、"
      f"10 回以下の字 {few}")
print(f"多い順に 100 種類の字が、全体の {top100:.1%} を占める")
print("例文の ID:", tok.encode("どうも仕方がない。"))
cap.save("char_freq.txt")

# ---- 語彙の中身の内訳（図の数値）---------------------------------------
kinds = Counter()
longest = []
for i in range(256, 256 + len(tok.merges)):
    try:
        s = tok.vocab[i].decode("utf-8")
        kinds[min(len(s), 5)] += 1
        longest.append((len(s), s))
    except UnicodeDecodeError:
        kinds[0] += 1
labels = {0: "文字の断片", 1: "1 字", 2: "2 字", 3: "3 字", 4: "4 字",
          5: "5 字以上"}
with open(OUT / "token_kinds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "label", "count"])
    for k in range(6):
        w.writerow([k, labels[k], kinds[k]])
        print(f"{labels[k]}: {kinds[k]}")
longest.sort(key=lambda x: -x[0])
print("長いトークン:", " / ".join(s.replace("\n", "\\n")
                                for _, s in longest[:8]))

# 学習データで実際に使われているトークン
train = np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16)
freq = np.bincount(train, minlength=tok.vocab_size)
print(f"train.bin: {len(train):,} トークン、"
      f"一度でも現れる ID は {int((freq > 0).sum())} 種類")
print("0〜255 のうち現れるもの:", int((freq[:256] > 0).sum()))
eot = tok.special_tokens["<|endoftext|>"]
for p in np.flatnonzero(train == eot):  # 文書の区切りの位置と、その前後
    near = train[max(0, p - 4):p + 5]
    around = "/".join(tok.token_str(int(i)) for i in near)
    print(f"<|endoftext|> の位置 {int(p):,}: {around!r}")
top = np.argsort(-freq)[:10]
print("多い順:", " ".join(f"{show(tok.vocab[int(i)])}({int(freq[i]):,})"
                       for i in top))
cap.save("vocab_stats.txt")
