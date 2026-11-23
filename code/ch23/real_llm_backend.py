"""同じ run_agent を、実在の小型 LLM（transformers）で動かす差し替え口。

Qwen3 の書式では、ツールは JSON スキーマで渡し、モデルは
<tool_call>{"name": ..., "arguments": {...}}</tool_call> を書く。
ここでは、それを本書の書式 <|tool|>式<|end|> に翻訳して run_agent に返す。
実行:  python ch23/real_llm_backend.py  （初回は約 1.5 GB を取得。GPU 推奨）
"""

import json
import re

from agent_loop import run_agent

# [snippet: tool-schema]
CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "整数の足し算・引き算・掛け算をする",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string",
                                          "description": "例: 37*48"}},
            "required": ["expression"],
        },
    },
}
# [end: tool-schema]


# [snippet: hf-backend]
class HFChatLLM:
    """本書の文脈文字列 ⇄ transformers の messages を翻訳する包み紙。"""

    def __init__(self, name: str = "Qwen/Qwen3-0.6B"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name)

    def __call__(self, context: str) -> str:
        messages = []  # <|役割|>本文<|end|> を 1 発話ずつ messages に
        for role, body in re.findall(r"<\|(\w+)\|>(.*?)<\|end\|>",
                                     context, re.S):
            if role == "tool":    # モデル自身の過去のツール呼び出し
                call = {"name": "calculator",
                        "arguments": {"expression": body}}
                messages.append({"role": "assistant", "content": "",
                                 "tool_calls": [{"type": "function",
                                                 "function": call}]})
            elif role == "result":  # 環境が返した結果
                messages.append({"role": "tool", "content": body})
            else:
                messages.append({"role": role, "content": body})
        # [snippet: hf-generate]
        ids = self.tok.apply_chat_template(
            messages, tools=[CALCULATOR_SCHEMA], add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True)
        out = self.model.generate(**ids, max_new_tokens=128, do_sample=False)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
        if m:  # ツール呼び出し → 本書の書式に直して返す
            args = json.loads(m.group(1))["arguments"]
            return f"<|tool|>{args['expression']}<|end|>"
        return f"<|assistant|>{text.strip()}<|end|>"
        # [end: hf-generate]
# [end: hf-backend]


if __name__ == "__main__":
    from tools import calculator
    llm = HFChatLLM()
    system = "計算は必ず calculator ツールで行ってください。"
    # [snippet: hf-run]
    def tolerant(expr):  # 「×」で書かれても受け付ける電卓
        return calculator(expr.replace("×", "*"))

    for q, tool in [("37×48は？", calculator), ("37×48は？", tolerant),
                    ("123456×789は？", tolerant)]:
        answer, context, n = run_agent(llm, q, tool=tool, system=system)
        print(context)
    # [end: hf-run]
