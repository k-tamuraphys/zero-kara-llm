"""エージェントが使うツール（電卓）と、ツールを使う計算問題の生成。

第 24 章もこのモジュールを使う。
  calculator("37*48") -> "1776"
  make_tool_example(rng, "mul") -> (messages, "1776")
"""

import ast
import operator
import random

# [snippet: calculator]
# 電卓が許す演算。これ以外（関数呼び出し、属性、名前など）は拒否する。
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul}


def _eval(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("使えない式です")


def calculator(expression: str) -> str:
    """整数の +, -, * だけを計算する電卓。失敗したら理由を文字列で返す。"""
    if len(expression) > 40:  # 巨大な式で計算機を止められないように
        return "エラー: 式が長すぎます"
    try:
        tree = ast.parse(expression, mode="eval")  # 文字列を構文木に
        return str(_eval(tree.body))
    except (SyntaxError, ValueError):
        return "エラー: 計算できない式です（使える記号は 0-9 + - * のみ）"
# [end: calculator]


# [snippet: tool-registry]
# ツールの名前 → (説明, 本体)。実際の API ではこれを JSON スキーマで渡す。
TOOLS = {
    "calculator": ("整数の足し算・引き算・掛け算をする。例: 37*48",
                   calculator),
}
# [end: tool-registry]


TEMPLATES = {
    "mul": ["{a}×{b}は？", "{a}掛ける{b}は？",
            "{a}と{b}を掛けるといくつですか？"],
    "two": ["{a}×{b}+{c}は？", "{a}×{b}に{c}を足すといくつですか？"],
}


def _num(rng, max_digits=3):
    """桁数を先に選んでから数を選ぶ（小さい数ばかりにならないように）。"""
    digits = rng.randint(1, max_digits)
    return rng.randint(10 ** (digits - 1) if digits > 1 else 0,
                       10 ** digits - 1)


# [snippet: tool-examples]
def make_tool_example(rng: random.Random, kind: str):
    """kind = "add"（加減算。自分で答える）、"mul"（掛け算）、
    "two"（掛けて足す）。後の二つは電卓を呼ぶ手本にする。"""
    if kind == "add":
        a, b = _num(rng), _num(rng)
        op = rng.choice("+-")
        a, b = (max(a, b), min(a, b)) if op == "-" else (a, b)
        q = f"{a}{op}{b}は？"
        answer = str(a + b if op == "+" else a - b)
        return [{"role": "user", "content": q},
                {"role": "assistant", "content": f"{answer}です。"}], answer
    # [snippet: tool-examples-call]
    a, b, c = _num(rng), _num(rng), _num(rng)
    q = rng.choice(TEMPLATES[kind]).format(a=a, b=b, c=c)
    expr = f"{a}*{b}" if kind == "mul" else f"{a}*{b}+{c}"
    answer = calculator(expr)
    return [{"role": "user", "content": q},
            {"role": "tool", "content": expr},       # モデルが書く呼び出し
            {"role": "result", "content": answer},   # 環境が差し込む結果
            {"role": "assistant", "content": f"{answer}です。"}], answer
    # [end: tool-examples-call]
# [end: tool-examples]
