"""実務のトークナイザ: OpenAI の tiktoken で GPT 系の BPE を使ってみる。

実行:  python ch07/tiktoken_demo.py   （code/ ディレクトリで実行する）
初回は語彙ファイルをダウンロードするので、ネットワーク接続が必要。
"""

import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ch07.capture import Capture
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer

cap = Capture()


def show(token: bytes) -> str:
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return f"<{token.hex(' ')}>"


# [snippet: tiktoken]
enc = tiktoken.get_encoding("gpt2")  # GPT-2 のトークナイザ
print("語彙の大きさ:", enc.n_vocab)
for text in ["Hello world", "tokenization", "12345678", "吾輩は猫である。"]:
    ids = enc.encode(text)
    cut = "/".join(show(enc.decode_single_token_bytes(i)) for i in ids)
    print(f"{len(ids):2d} トークン  {cut}")
print(enc.decode(enc.encode("Hello world")))
# [end: tiktoken]
cap.save("tiktoken_gpt2.txt")

# 同じ文を、時代の違うトークナイザで数える（表の数値）
ours = BPETokenizer.load(DATA_DIR / "bpe4096.json")
texts = {
    "ja": "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "en": "I am a cat. As yet I have no name. "
          "I have no idea where I was born.",
    "num": "1234567890",
}
for key, text in texts.items():
    row = [f"{key}: {len(text)} 字, {len(text.encode('utf-8'))} バイト"]
    row.append(f"本書 {len(ours.encode(text))}")
    for name in ["gpt2", "cl100k_base", "o200k_base"]:
        enc = tiktoken.get_encoding(name)
        row.append(f"{name} {len(enc.encode(text))}")
    print(" | ".join(row))
for name in ["gpt2", "cl100k_base", "o200k_base"]:
    enc = tiktoken.get_encoding(name)
    ids = enc.encode("1234567890")
    cut = "/".join(show(enc.decode_single_token_bytes(i)) for i in ids)
    print(f"{name}: 語彙 {enc.n_vocab:,}  {cut}")
enc = tiktoken.get_encoding("o200k_base")
for text in ["吾輩は猫である。", " world", "world", "strawberry"]:
    ids = enc.encode(text)
    cut = "/".join(show(enc.decode_single_token_bytes(i)) for i in ids)
    print(f"o200k_base {text!r}: {ids} {cut}")
try:
    enc.encode("<|endoftext|>")
except ValueError as e:
    print("ValueError:", str(e).split("\n")[0][:60])
print(enc.encode("<|endoftext|>", allowed_special="all"))
cap.save("tiktoken_compare.txt")
