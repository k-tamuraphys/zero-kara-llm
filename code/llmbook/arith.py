"""計算ドリル: 足し算・引き算の問題を自動生成し、モデルの正答率を測る。

事前学習では「12+35=47」のような行を小説と混ぜて学習させる。
事後学習・強化学習・エージェントの章では、同じ問題を題材に使う。
"""

import random

import torch


# [snippet: make-problem]
def make_problem(rng: random.Random, max_value: int = 999,
                 ops: str = "+-") -> tuple[str, str]:
    """問題文（例 "12+35="）と答え（例 "47"）の組を 1 つ作る。"""
    op = rng.choice(ops)
    # 桁数を先に選ぶ。こうしないと 3 桁の問題ばかりになってしまう。
    a = rng.randint(0, min(max_value, 10 ** rng.randint(1, 3) - 1))
    b = rng.randint(0, min(max_value, 10 ** rng.randint(1, 3) - 1))
    if op == "-" and a < b:
        a, b = b, a  # 答えが負にならないようにする
    answer = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"{a}{op}{b}=", str(answer)
# [end: make-problem]


# [snippet: drill-text]
def drill_text(n_lines: int, seed: int = 0, max_value: int = 999,
               ops: str = "+-") -> str:
    """"12+35=47" の形の行を n_lines 行ならべたテキストを作る。"""
    rng = random.Random(seed)
    lines = []
    for _ in range(n_lines):
        question, answer = make_problem(rng, max_value, ops)
        lines.append(question + answer)
    return "\n".join(lines) + "\n"
# [end: drill-text]


# [snippet: complete]
@torch.no_grad()
def complete(model, tok, prompt: str, max_new_tokens: int = 8,
             stop: str = "\n", device: str = "cpu") -> str:
    """prompt の続きを貪欲法（温度 0）で生成し、stop の手前までを返す。"""
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], device=device)
    stop_id = tok.encode(stop)[0]
    out = model.generate(idx, max_new_tokens, temperature=0.0,
                         stop_token=stop_id)
    text = tok.decode(out[0, idx.size(1):].tolist())
    return text.split(stop)[0]
# [end: complete]


# [snippet: drill-accuracy]
def drill_accuracy(model, tok, n_problems: int = 200, seed: int = 12345,
                   max_value: int = 999, ops: str = "+-",
                   device: str = "cpu") -> float:
    """"a+b=" を与えて続きを生成させ、正しい答えが出る割合を返す。"""
    rng = random.Random(seed)
    n_correct = 0
    for _ in range(n_problems):
        question, answer = make_problem(rng, max_value, ops)
        output = complete(model, tok, "\n" + question, device=device)
        n_correct += output == answer
    return n_correct / n_problems
# [end: drill-accuracy]
