"""文字単位のトークナイザと UTF-8 のバイト列を確かめる。

実行:  python ch07/units.py   （code/ ディレクトリで実行する）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ch07.capture import Capture
from llmbook.corpus import DATA_DIR, load_corpus
from llmbook.tokenizer import BPETokenizer, CharTokenizer

cap = Capture()
text = load_corpus()  # 漱石 14 作品

# [snippet: char-demo]
tok = CharTokenizer(text)  # コーパスに現れる文字に番号を振る
print("語彙の大きさ:", tok.vocab_size)
ids = tok.encode("吾輩は猫である。")
print(ids)
print(tok.decode(ids))
try:
    tok.encode("12+35=47")  # 「+」は漱石の小説に一度も現れない
except KeyError as e:
    print("KeyError:", e)
# [end: char-demo]
cap.save("char_demo.txt")

# [snippet: utf8]
for ch in ["A", "é", "あ", "吾", "\U0001F600"]:  # 最後は笑顔の絵文字
    raw = ch.encode("utf-8")  # 文字列 → バイト列
    print(f"U+{ord(ch):04X}  {len(raw)} バイト  {raw.hex(' '):12s}{list(raw)}")
# [end: utf8]
cap.save("utf8.txt")

# [snippet: bytes-roundtrip]
raw = "吾輩".encode("utf-8")
print(list(raw))  # 6 個の整数
print(raw.decode("utf-8"))  # バイト列 → 文字列
try:
    raw[:4].decode("utf-8")  # 2 文字目の途中で切ってみる
except UnicodeDecodeError as e:
    print("UnicodeDecodeError:", e.reason)
# [end: bytes-roundtrip]
cap.save("bytes_roundtrip.txt")

# コーパス全体を 3 通りの単位で数える（表の数値）
bpe = BPETokenizer.load(DATA_DIR / "bpe4096.json")
n_char = len(text)
n_byte = len(text.encode("utf-8"))
n_bpe = len(bpe.encode(text))
print(f"文字    : 語彙 {tok.vocab_size:5d}  列の長さ {n_char:,}")
print(f"バイト  : 語彙 {256:5d}  列の長さ {n_byte:,}"
      f"（文字の {n_byte / n_char:.2f} 倍）")
print(f"BPE     : 語彙 {bpe.vocab_size:5d}  列の長さ {n_bpe:,}"
      f"（1 トークン {n_char / n_bpe:.2f} 字）")
print(f"文脈長 128 トークンで見える範囲: 文字なら 128 字、"
      f"バイトなら {128 * n_char / n_byte:.0f} 字、"
      f"BPE なら {128 * n_char / n_bpe:.0f} 字")
cap.save("units.txt")
