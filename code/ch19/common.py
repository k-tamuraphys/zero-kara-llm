"""第 19 章の共通部品: モデルの読み込みと、評価用の問題集。"""

import random
import re

import torch

from llmbook.chat import QUESTION_TEMPLATES, make_chat_example, render_chat
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.train import load_checkpoint

torch.set_num_threads(2)  # CPU を使いすぎないようにする
CKPT_DIR = DATA_DIR.parent / "checkpoints"


def load(name: str = "sft"):
    """チェックポイント（"sft" → checkpoints/sft.pt）とトークナイザを返す。"""
    tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
    model = load_checkpoint(CKPT_DIR / f"{name}.pt")
    model.eval()
    return model, tok


def has_carry(a: int, b: int, op: str) -> bool:
    """筆算で繰り上がり（引き算なら繰り下がり）が 1 回でも起きるか。"""
    while a > 0 or b > 0:
        da, db = a % 10, b % 10
        if (op == "+" and da + db >= 10) or (op == "-" and da < db):
            return True
        a, b = a // 10, b // 10
    return False


# [snippet: make-problems]
def make_problems(n: int, seed: int) -> list[dict]:
    """評価用の問題を n 問作る。質問文・正解のほかに、分類用の情報を付ける。"""
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        messages, answer = make_chat_example(rng)  # 学習データと同じ作り方
        question = messages[0]["content"]
        a, b = (int(s) for s in re.findall(r"\d+", question))
        op = "+" if ("+" in question or "足" in question) else "-"
        problems.append({
            "question": question,          # 例 "12+35は？"
            "target": f"{answer}です。",    # 模範応答
            "op": op,                      # 演算
            "digits": len(str(max(a, b))),  # 大きいほうの桁数
            "carry": has_carry(a, b, op),  # 繰り上がり・繰り下がりの有無
            "a": a, "b": b,
        })
    return problems
# [end: make-problems]


def template_id(p: dict) -> int:
    """問題 p の質問文が、3 通りの言い回しの何番目かを返す。"""
    for i, t in enumerate(QUESTION_TEMPLATES[p["op"]]):
        if t.format(a=p["a"], b=p["b"]) == p["question"]:
            return i
    return -1


def chat_prompt_ids(tok, question: str) -> list[int]:
    """質問を対話形式のプロンプト（トークン ID の列）にする。"""
    return tok.encode(render_chat([{"role": "user", "content": question}],
                                  add_generation_prompt=True))


def evaluate(name: str, n: int, seed: int, batch: int = 64) -> list[dict]:
    """モデル name に問題集 (n, seed) を解かせ、応答と正誤を付けて返す。

    後半で作る高速なエンジン（engine.py）を使う。結果は out/cache に保存し、
    2 回目からは読み直すだけにする。
    """
    import json
    from pathlib import Path
    from engine import Engine, reply_batch
    cache = Path(__file__).parent / "out" / "cache" / f"{name}_{seed}_{n}.json"
    problems = make_problems(n, seed)
    if cache.exists():
        replies = json.loads(cache.read_text())
    else:
        model, tok = load(name)
        engine = Engine(model, n_slots=batch)
        replies = []
        for i in range(0, n, batch):
            qs = [p["question"] for p in problems[i:i + batch]]
            replies += reply_batch(engine, tok, qs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(replies, ensure_ascii=False))
    for p, r in zip(problems, replies):
        p["reply"], p["correct"] = r, r == p["target"]
    return problems


def answer_all(model, tok, questions: list[str], batch: int = 64) -> list[str]:
    """質問のリストにまとめて応答させる（エンジンを使う。キャッシュなし）。"""
    from engine import Engine, reply_batch
    engine = Engine(model, n_slots=batch)
    replies = []
    for i in range(0, len(questions), batch):
        replies += reply_batch(engine, tok, questions[i:i + batch])
    return replies
