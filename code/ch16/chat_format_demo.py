"""対話 1 件が、どんなトークン列と正解ラベルになるかを表示する。

実行:  python ch16/chat_format_demo.py
出力:  ch16/out/chat_encode.txt, ch16/out/chat_collate.txt（数秒）
"""

import sys

from common import OUT_DIR, tok
from llmbook.chat import IGNORE, collate, encode_chat_example, render_chat

sys.stdout = open(OUT_DIR / "chat_encode.txt", "w")  # 結果をファイルへ
# [snippet: demo-encode]
messages = [{"role": "user", "content": "12+35は？"},
            {"role": "assistant", "content": "47です。"}]
print(render_chat(messages))
ids, labels = encode_chat_example(tok, messages)
for t, (i, y) in enumerate(zip(ids, labels)):
    target = "（学習しない）" if y == IGNORE else repr(tok.decode([y]))
    print(f"{t:2d}: {tok.decode([i])} -> {target}")
# [end: demo-encode]

sys.stdout = open(OUT_DIR / "chat_collate.txt", "w")
# [snippet: demo-collate]
long = [{"role": "user", "content": "7と8を足すといくつですか？"},
        {"role": "assistant", "content": "15です。"}]
x, y = collate([encode_chat_example(tok, messages),
                encode_chat_example(tok, long)],
               pad_id=tok.special_tokens["<|pad|>"])
print("x.shape =", tuple(x.shape))  # (B, T) = (2, 最長の例の長さ)
for b in range(2):  # 1 行に収まるよう、例ごとに表示する
    print(f"x[{b}] =", x[b].tolist())
    print(f"y[{b}] =", y[b].tolist())
# [end: demo-collate]
