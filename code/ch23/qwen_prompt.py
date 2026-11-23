"""Qwen3 に実際に渡るプロンプトを表示する（モデル本体は読み込まない）。

apply_chat_template に tools= で JSON スキーマを渡すと、ツールの定義が
システムプロンプトの位置に文字列として埋め込まれることを確かめる。
実行:  python ch23/qwen_prompt.py  > ch23/out/qwen_prompt.txt  （数秒）
"""

from transformers import AutoTokenizer

from real_llm_backend import CALCULATOR_SCHEMA

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# [snippet: qwen-prompt]
messages = [
    {"role": "system",
     "content": "計算は必ず calculator ツールで行ってください。"},
    {"role": "user", "content": "37×48は？"},
]
prompt = tok.apply_chat_template(
    messages, tools=[CALCULATOR_SCHEMA], add_generation_prompt=True,
    enable_thinking=False, tokenize=False)  # トークン列でなく文字列で返す
# [end: qwen-prompt]
# 紙面の節約のため、空行だけを除いて表示する（文字は変えない）
print("\n".join(line for line in prompt.split("\n") if line.strip()))
