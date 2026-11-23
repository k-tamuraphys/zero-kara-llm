"""トークナイザ: 文字単位 (CharTokenizer) とバイト単位 BPE (BPETokenizer)。"""

import json
import re
from pathlib import Path

import numpy as np

# [snippet: special-tokens]
# 事後学習以降で使う特殊トークン。語彙の末尾にあらかじめ席を確保しておく。
SPECIAL_TOKENS = [
    "<|endoftext|>",  # 文書の区切り
    "<|user|>",  # ユーザの発話の開始
    "<|assistant|>",  # アシスタントの発話の開始
    "<|end|>",  # 発話の終了
    "<|tool|>",  # ツール呼び出しの開始
    "<|result|>",  # ツール実行結果の開始
    "<|think|>",  # 思考の開始
    "<|pad|>",  # 長さをそろえるための詰め物
]
# [end: special-tokens]


# [snippet: char-tokenizer]
class CharTokenizer:
    """1 文字を 1 トークンとする最も単純なトークナイザ。"""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    # [snippet: bpe-encode]
    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    # [snippet: bpe-decode]
    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)
# [end: char-tokenizer]


# [snippet: count-pairs]
def count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    """隣り合うトークンの組が何回現れるかを数える。"""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
# [end: count-pairs]


# [snippet: merge-pair]
def merge_pair(ids: list[int], pair: tuple[int, int],
               new_id: int) -> list[int]:
    """ids の中の pair をすべて new_id に置き換える。"""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
# [end: merge-pair]


def _merge_pair_np(ids: np.ndarray, a: int, b: int, new_id: int) -> np.ndarray:
    """merge_pair と同じ処理を NumPy でまとめて行う（大きな配列用）。"""
    if len(ids) < 2:
        return ids
    # [end: bpe-encode]
    match = (ids[:-1] == a) & (ids[1:] == b)
    if not match.any():
        return ids
    if a == b:
        # "aaa" のように重なる場合は左から順に 1 つおきに採用する。
        pos = np.arange(len(match))
        prev = np.concatenate(([False], match[:-1]))
        run_start = np.where(match & ~prev, pos, 0)
        run_start = np.maximum.accumulate(run_start)
        match &= (pos - run_start) % 2 == 0
    keep = np.ones(len(ids), dtype=bool)
    keep[1:][match] = False  # 組の 2 つ目のトークンを消す
    out = ids.copy()
    out[:-1][match] = new_id  # 組の 1 つ目を新しいトークンにする
    return out[keep]


class BPETokenizer:
    """バイト単位の BPE (byte pair encoding) トークナイザ。

    ID の割り当て:
      0 .. 255              : 1 バイトそのもの
      256 .. 256+マージ数-1  : マージで作られたトークン
      その後ろ               : 特殊トークン
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # ---- 学習 ----------------------------------------------------------
    # [snippet: bpe-train]
    def train(self, text: str, vocab_size: int, special_tokens=SPECIAL_TOKENS,
              verbose: bool = False) -> None:
        """text から、語彙が vocab_size 個になるまでマージ規則を学習する。"""
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges >= 0
        raw = text.encode("utf-8")
        ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        size = 256 + n_merges  # 特殊トークンを除く語彙の大きさ
        for k in range(n_merges):
            # 組 (a, b) を 1 つの整数 a*size+b で表し、出現回数を数える。
            pair_code = ids[:-1] * size + ids[1:]
            counts = np.bincount(pair_code, minlength=size * size)
            best = int(counts.argmax())  # 最も多い組
            a, b = divmod(best, size)
            new_id = 256 + k
            ids = _merge_pair_np(ids, a, b, new_id)
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            if verbose and (k < 10 or (k + 1) % 500 == 0):
                token = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"マージ {k + 1:4d}: ({a}, {b}) -> {new_id} "
                      f"[{token}] {int(counts[best]):,} 回")
        for i, name in enumerate(special_tokens):
            self.special_tokens[name] = 256 + n_merges + i
    # [end: bpe-train]

    # ---- 符号化 --------------------------------------------------------
    # [snippet: bpe-encode-chunk]
    def _encode_chunk(self, text: str) -> list[int]:
        """特殊トークンを含まない文字列をトークン ID 列にする。"""
        raw = text.encode("utf-8")
        if len(raw) > 4096:  # 長い文字列は NumPy でまとめて処理する
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
            for (a, b), new_id in self.merges.items():  # 学習した順に適用
                arr = _merge_pair_np(arr, a, b, new_id)
            return arr.tolist()
        ids = list(raw)
        while len(ids) >= 2:
            # 現在の列に含まれる組のうち、最も早く学習されたものを探す。
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # もうマージできる組がない
            ids = merge_pair(ids, pair, self.merges[pair])
        return ids
    # [end: bpe-encode-chunk]

    def encode(self, text: str) -> list[int]:
        """文字列をトークン ID 列にする。特殊トークンは 1 トークンになる。"""
        if not self.special_tokens:
            return self._encode_chunk(text)
        specials = "|".join(re.escape(s) for s in self.special_tokens)
        pattern = "(" + specials + ")"  # 括弧で囲むと区切り自身も残る
        ids: list[int] = []
        for piece in re.split(pattern, text):
            if piece in self.special_tokens:
                ids.append(self.special_tokens[piece])
            elif piece:
                ids.extend(self._encode_chunk(piece))
        return ids

    # ---- 復号 ----------------------------------------------------------
    def decode(self, ids) -> str:
        """トークン ID 列を文字列に戻す。"""
        id_to_special = {i: s for s, i in self.special_tokens.items()}
        parts = []
        for i in ids:
            i = int(i)
            if i in id_to_special:
                parts.append(id_to_special[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        # 文字の途中で切れたバイト列は「?」のような置換文字になる。
        return b"".join(parts).decode("utf-8", errors="replace")
    # [end: bpe-decode]

    def token_str(self, i: int) -> str:
        """トークン 1 個を表示用の文字列にする。"""
        return self.decode([i])

    # ---- 保存と読み込み ------------------------------------------------
    def save(self, path) -> None:
        data = {
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        for a, b, new_id in data["merges"]:
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = data["special_tokens"]
        return tok
