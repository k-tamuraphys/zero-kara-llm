"""実務での書き方: transformers のチャットテンプレート (apply_chat_template)。

Qwen3 のトークナイザ（約 16 MB。モデルの重みは取得しない）をダウンロードし、
本書の render_chat と同じことを実在の形式 (ChatML) で行う。
実行:  python ch16/hf_chat_template.py      出力: ch16/out/hf_chat_template.txt
"""

import jinja2

from common import OUT_DIR
from llmbook.chat import render_chat

lines = []
# [snippet: hf-template]
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
messages = [
    {"role": "system", "content": "あなたは計算アシスタントです。"},
    {"role": "user", "content": "12+35は？"},
]
# 学習データを作るときは add_generation_prompt=False で、応答まで含める。
# 推論のときは True にして、「ここからアシスタントが書く」印を末尾に付ける。
text = tokenizer.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                    enable_thinking=False)["input_ids"]
# [end: hf-template]
lines.append(text)
lines.append(f"トークン数 {len(ids)}、先頭 2 トークン: "
             f"{tokenizer.convert_ids_to_tokens(ids[:2])}")

# [snippet: jinja-template]
# 本書の形式を、transformers と同じ Jinja 記法のテンプレートで書く
BOOK_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}")
rendered = jinja2.Template(BOOK_TEMPLATE).render(
    messages=messages, add_generation_prompt=True)
assert rendered == render_chat(messages, add_generation_prompt=True)
# [end: jinja-template]
lines.append(rendered)

with open(OUT_DIR / "hf_chat_template.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
