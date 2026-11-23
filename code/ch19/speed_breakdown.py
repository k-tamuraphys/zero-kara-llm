"""1,000 問の評価が速くなった内訳: KV キャッシュの効果と、バッチの効果。

check_engine.py は「素朴な生成」と「エンジンで 64 問ずつ」の両端だけを比べた。
ここでは、間に「エンジンで 1 問ずつ」をはさみ、同じ実行の中で 3 通りを測る。
"""

import time

from common import load, make_problems
from engine import Engine, reply_batch
from llmbook.chat import chat_reply

model, tok = load("sft")
questions = [p["question"] for p in make_problems(1000, seed=777)]


def timed(batch: int) -> tuple[float, list[str]]:
    """エンジンで batch 問ずつ解かせ、(かかった秒数, 応答) を返す。"""
    engine = Engine(model, n_slots=batch)
    t0, replies = time.perf_counter(), []
    for i in range(0, len(questions), batch):
        replies += reply_batch(engine, tok, questions[i:i + batch])
    return time.perf_counter() - t0, replies


t0 = time.perf_counter()
plain = [chat_reply(model, tok, q) for q in questions]
t_plain = time.perf_counter() - t0
t_one, r_one = min(timed(1) for _ in range(2))
t_64, r_64 = min(timed(64) for _ in range(3))
assert plain == r_one == r_64  # 応答は 3 通りとも同じ

print("1,000 問（種 777）を解く時間")
print(f"  素朴な生成（キャッシュなし、1 問ずつ）: {t_plain:5.1f} 秒")
print(f"  エンジン（キャッシュあり、1 問ずつ）  : {t_one:5.1f} 秒"
      f"（キャッシュで {t_plain / t_one:.1f} 倍）")
print(f"  エンジン（キャッシュあり、64 問ずつ） : {t_64:5.1f} 秒"
      f"（バッチでさらに {t_one / t_64:.1f} 倍、"
      f"合わせて {t_plain / t_64:.1f} 倍）")
