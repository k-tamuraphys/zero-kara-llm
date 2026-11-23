"""ch23_tool_sft.pt が掛け算を間違えた軌跡を集めて、失敗の型を数える。

実行:  python ch23/failures.py  > ch23/out/failures.txt      （約 1 分）
"""

import random
from collections import Counter

import torch

from agent_loop import run_agent
from common import CKPT_DIR, device
from mini_agent import MiniGPTLLM
from tools import make_tool_example
from llmbook.train import load_checkpoint

torch.manual_seed(0)
llm = MiniGPTLLM(load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt", device))
rng = random.Random(12345)  # mini_agent.py の評価と同じ 100 問
kinds, shown = Counter(), 0
asked, wrong = Counter(), Counter()  # 言い回しごとの問題数と誤答数
for _ in range(100):
    messages, answer = make_tool_example(rng, "mul")
    question = messages[0]["content"]
    style = ("a×bは？" if "×" in question else
             "a掛けるbは？" if "は？" in question else
             "aとbを掛けるといくつですか？")
    asked[style] += 1
    reply, context, n_calls = run_agent(llm, question)
    if reply == f"{answer}です。":
        continue
    wrong[style] += 1
    expr = messages[1]["content"]
    called = (context.split("<|tool|>")[1].split("<|end|>")[0]
              if n_calls else "")
    kind = ("式の写し間違い" if n_calls and called != expr else
            "結果の写し間違い" if n_calls else "電卓を呼ばない")
    kinds[kind] += 1
    if shown < 4:
        print(f"[{kind}] {context}")
        shown += 1
print(dict(kinds))
print("言い回しごとの誤答数:")
for style, n in asked.items():
    print(f"  {style}: {n} 問中 {wrong[style]} 問")
