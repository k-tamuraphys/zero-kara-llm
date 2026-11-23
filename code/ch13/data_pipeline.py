"""事前学習データの整形・品質フィルタ・重複除去・memmap を小さく実演する。

実行: python ch13/data_pipeline.py  （約 10 秒）
出力: ch13/out/data_pipeline.txt
"""

import random
import re
import zlib
from collections import Counter

import numpy as np

from llmbook.arith import drill_text
from llmbook.corpus import DATA_DIR, clean_aozora
from llmbook.tokenizer import BPETokenizer

tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")

print("== 1. 青空文庫形式の整形 ==")
# [snippet: clean-demo]
raw = ("こころ\n夏目漱石\n\n-------------------------------------------\n"
       "【テキスト中に現れる記号について】\n《》：ルビ\n"
       "-------------------------------------------\n"
       "　私《わたくし》はその人を常に先生と呼んでいた。［＃「先生」に傍点］\n"
       "\n\n\n底本：「こころ」角川文庫\n")
print(repr(clean_aozora(raw)))
# [end: clean-demo]

print("== 2. 品質フィルタ（経験則） ==")


# [snippet: quality]
def quality_ok(line: str) -> bool:
    """短すぎる行、同じ文字の繰り返し、記号ばかりの行を落とす。"""
    if len(line) < 10:
        return False
    if Counter(line).most_common(1)[0][1] / len(line) > 0.3:
        return False  # 1 種類の文字が 3 割を超える
    kana_kanji = len(re.findall(r"[ぁ-んァ-ン一-龥]", line))
    return kana_kanji / len(line) > 0.5  # 日本語の文字が半分以上
# [end: quality]


for line in ["　私はその人を常に先生と呼んでいた。",
             "ああああああああああああああ！！",
             "https://example.com/?id=12345&ref=top",
             "続きを読む"]:
    print(f"{'残す' if quality_ok(line) else '落とす'}: {line}")

print("== 3. 完全一致の重複除去（計算ドリル 12 万行） ==")
# [snippet: exact-dedup]
lines = drill_text(120_000, seed=0).splitlines()
unique = list(dict.fromkeys(lines))  # 出てきた順を保ったまま重複を除く
print(f"{len(lines):,} 行 → 重複を除くと {len(unique):,} 行")
top = Counter(lines).most_common(3)
print("最も多い行:", ", ".join(f"{s} ×{n}" for s, n in top))
# [end: exact-dedup]

print("== 4. ほぼ重複の検出（文字 5-gram の Jaccard 係数と MinHash） ==")


# [snippet: minhash]
def shingles(text: str, n: int = 5) -> set[str]:
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)

def minhash(s: set, n_hash: int = 128, seed: int = 0) -> list[int]:
    """塩（salt）を変えた n_hash 個のハッシュ関数ごとに、最小値を取る。"""
    rng = random.Random(seed)
    salts = [rng.getrandbits(32) for _ in range(n_hash)]
    return [min(zlib.crc32(f"{salt}:{x}".encode()) for x in s)
            for salt in salts]

def minhash_sim(m1: list[int], m2: list[int]) -> float:
    return sum(a == b for a, b in zip(m1, m2)) / len(m1)  # 一致した割合
# [end: minhash]


docs = {
    "A": "私はその人を常に先生と呼んでいた。だからここでもただ先生と書く"
         "だけで本名は打ち明けない。",
    "B": "私はその人を常に先生と呼んでいた。だからここでも先生と書くだけ"
         "で本名は打ち明けない。",  # A から「ただ」を消した転載
    "C": "山路を登りながら、こう考えた。智に働けば角が立つ。",
}
sh = {k: shingles(v) for k, v in docs.items()}
mh = {k: minhash(v) for k, v in sh.items()}
for a, b in [("A", "B"), ("A", "C")]:
    print(f"{a}-{b}: Jaccard {jaccard(sh[a], sh[b]):.3f} / "
          f"MinHash 推定 {minhash_sim(mh[a], mh[b]):.3f}")

print("== 5. トークン列を uint16 で保存し、memmap で読む ==")
# [snippet: memmap]
data = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
print(f"トークン数 {len(data):,} / ファイル {data.nbytes:,} バイト")
eot = tok.special_tokens["<|endoftext|>"]
cut = int(np.flatnonzero(data == eot)[0])  # 最初の文書区切り
print(f"小説 {cut:,} トークン / ドリル {len(data) - cut - 2:,} トークン")
print("区切りの前後:", repr(tok.decode(data[cut - 9:cut + 9].tolist())))
# [end: memmap]
