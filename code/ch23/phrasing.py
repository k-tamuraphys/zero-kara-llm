"""加減算の質問の「言い回し」を変えると、エージェントはどう振る舞うか。

SFT の手本（tools.py）の加減算は「a+bは？」型だけだった。第 16 章の 3 通りの
言い回し（llmbook.chat.QUESTION_TEMPLATES）で、電卓の使用率と正答率を測る。
実行:  python ch23/phrasing.py  > ch23/out/phrasing.txt        （約 2 分）
"""

import random

import torch

from agent_loop import run_agent
from common import CKPT_DIR, device, tok
from mini_agent import MiniGPTLLM
from llmbook.arith import make_problem
from llmbook.chat import QUESTION_TEMPLATES, chat_accuracy
from llmbook.train import load_checkpoint

torch.manual_seed(0)
llm = MiniGPTLLM(load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt", device))
n_problems = 150
examples = {True: [], False: []}  # 電卓を呼んだ／呼ばなかった誤答の軌跡
print(f"加減算（各 {n_problems} 問）: 言い回しごとの電卓使用率と正答率")
# [snippet: phrasing]
for style in range(3):  # 0:「a+bは？」 1:「a足すbは？」 2:「aとbを足すと…」
    rng = random.Random(12345)  # 3 通りとも同じ数の組で比べる
    n = {True: 0, False: 0}     # 電卓を呼んだ問題数／呼ばなかった問題数
    ok = {True: 0, False: 0}    # そのうち正解した数
    for _ in range(n_problems):
        problem, answer = make_problem(rng, ops="+-")  # 例 "64+6=", "70"
        op = "+" if "+" in problem else "-"
        a, b = problem.rstrip("=").split(op)
        question = QUESTION_TEMPLATES[op][style].format(a=a, b=b)
        reply, context, n_calls = run_agent(llm, question)
        called, correct = n_calls > 0, reply == f"{answer}です。"
        n[called] += 1
        ok[called] += correct
        # [end: phrasing]
        if style > 0 and not correct:
            examples[called].append(context)
    names = "／".join(QUESTION_TEMPLATES[op][style] for op in "+-")
    print(f"■ {names}")
    print(f"  電卓使用 {n[True] / n_problems:5.1%}"
          f"  正答率 {(ok[True] + ok[False]) / n_problems:5.1%}"
          f"（呼んだ {n[True]} 問中 {ok[True]} 問、"
          f"呼ばない {n[False]} 問中 {ok[False]} 問が正解）")
print("電卓を呼んで間違えた軌跡の例:")
for context in examples[True][:2] + examples[True][-1:]:
    print(" ", context)
print("電卓を呼ばずに間違えた軌跡の例:")
for context in examples[False][:2]:
    print(" ", context)
# 電卓を使わせず、<|assistant|> を置いて直接答えさせた場合
# （第 16 章と同じ測り方）
print("直接答えさせたときの加減算の正答率（3 通りの言い回しの混合、200 問）:")
for name in ["sft.pt", "ch23_tool_sft.pt"]:
    model = load_checkpoint(CKPT_DIR / name, device)
    acc = chat_accuracy(model, tok, ops="+-", device=device)
    print(f"  {name:17s} {acc:.1%}")
