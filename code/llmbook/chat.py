"""対話形式と「計算アシスタント」の学習データ。

本書の対話形式では、すべての発話を  <|役割|>本文<|end|>  の形で書く。
  <|user|>12+35は？<|end|><|assistant|>47です。<|end|>
"""

import random

import torch

from .arith import make_problem

IGNORE = -100  # 損失を計算しない位置の目印（F.cross_entropy の ignore_index）

# [snippet: templates]
# 同じ計算をたずねるいろいろな言い方。{a}, {b} に数が入る。
QUESTION_TEMPLATES = {
    "+": ["{a}+{b}は？", "{a}足す{b}は？", "{a}と{b}を足すといくつですか？"],
    "-": ["{a}-{b}は？", "{a}引く{b}は？", "{a}から{b}を引くといくつですか？"],
    "*": ["{a}×{b}は？", "{a}掛ける{b}は？",
          "{a}と{b}を掛けるといくつですか？"],
}
# [end: templates]


# [snippet: render-chat]
def render_chat(messages: list[dict],
                add_generation_prompt: bool = False) -> str:
    """発話のリストを 1 本の文字列にする。

    messages の各要素は {"role": "user", "content": "12+35は？"} の形。
    add_generation_prompt=True なら末尾に <|assistant|> を付け、
    モデルに「ここからアシスタントの発話を書け」と伝える。
    """
    text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
    if add_generation_prompt:
        text += "<|assistant|>"
    return text
# [end: render-chat]


# [snippet: make-chat-example]
def make_chat_example(rng: random.Random, max_value: int = 999,
                      ops: str = "+-") -> tuple[list[dict], str]:
    """質問と模範応答からなる対話を 1 つ作る。答えの数字も返す。"""
    problem, answer = make_problem(rng, max_value, ops)  # 例 "12+35=", "47"
    op = next(ch for ch in problem if ch in "+-*")
    a, b = problem.rstrip("=").split(op)
    question = rng.choice(QUESTION_TEMPLATES[op]).format(a=a, b=b)
    messages = [{"role": "user", "content": question},
                {"role": "assistant", "content": f"{answer}です。"}]
    return messages, answer
# [end: make-chat-example]


# [snippet: encode-chat-example]
def encode_chat_example(tok, messages: list[dict], learn_role: bool = False
                        ) -> tuple[list[int], list[int]]:
    """対話をトークン列 ids と、損失用の正解 labels にする。

    labels[t] は位置 t の「次のトークン」。モデルが書くべき発話（user と
    result 以外）の本文だけを正解にし、それ以外は IGNORE にする。
    learn_role=True なら役割トークン（<|assistant|> や <|tool|>）も正解に
    含める。モデル自身に「次に何をするか」を選ばせたいときに使う。
    """
    ids: list[int] = []
    is_target: list[bool] = []
    for m in messages:
        head = tok.encode(f"<|{m['role']}|>")
        body = tok.encode(m["content"] + "<|end|>")
        ids += head + body
        learn = m["role"] not in ("user", "result")
        is_target += [learn and learn_role] * len(head) + [learn] * len(body)
    labels = [ids[t + 1] if is_target[t + 1] else IGNORE
              for t in range(len(ids) - 1)]
    return ids[:-1], labels
# [end: encode-chat-example]


# [snippet: collate]
def collate(examples: list[tuple[list[int], list[int]]], pad_id: int):
    """長さの違う例を、右側を詰め物で埋めて 1 つのバッチにまとめる。"""
    T = max(len(ids) for ids, _ in examples)
    x = torch.full((len(examples), T), pad_id, dtype=torch.long)
    y = torch.full((len(examples), T), IGNORE, dtype=torch.long)
    for i, (ids, labels) in enumerate(examples):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(labels)] = torch.tensor(labels)
    return x, y
# [end: collate]


# [snippet: chat-reply]
@torch.no_grad()
def chat_reply(model, tok, question: str, max_new_tokens: int = 24,
               temperature: float = 0.0, device: str = "cpu") -> str:
    """質問を対話形式にして与え、アシスタントの応答を生成して返す。"""
    model.eval()
    prompt = render_chat([{"role": "user", "content": question}],
                         add_generation_prompt=True)
    idx = torch.tensor([tok.encode(prompt)], device=device)
    end_id = tok.special_tokens["<|end|>"]
    out = model.generate(idx, max_new_tokens, temperature=temperature,
                         stop_token=end_id)
    reply = tok.decode(out[0, idx.size(1):].tolist())
    return reply.split("<|end|>")[0]
# [end: chat-reply]


# [snippet: chat-accuracy]
def chat_accuracy(model, tok, n_problems: int = 200, seed: int = 12345,
                  max_value: int = 999, ops: str = "+-",
                  device: str = "cpu") -> float:
    """対話形式で質問し、応答が模範応答「◯◯です。」と一致する割合を返す。"""
    rng = random.Random(seed)
    n_correct = 0
    for _ in range(n_problems):
        messages, answer = make_chat_example(rng, max_value, ops)
        reply = chat_reply(model, tok, messages[0]["content"], device=device)
        n_correct += reply == f"{answer}です。"
    return n_correct / n_problems
# [end: chat-accuracy]
