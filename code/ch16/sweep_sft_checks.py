"""計算アシスタント ch16_sft.pt について、本文の問いを二つ確かめる。

(1) 評価用の 200 問のうち、学習データと同じ質問の 26 問を除くと、
    対話の正答率はどう変わるか（丸暗記で当たっていないか）。
(2) SFT の後に、ドリル形式「⏎12+35=」でたずねると何を書くか。
実行:  python ch16/sweep_sft_checks.py > ch16/out/sweep_sft_checks.txt
CPU で約 10 秒。
"""

import random

from common import CKPT_DIR, device, tok
from llmbook.arith import complete
from llmbook.chat import chat_reply, make_chat_example
from llmbook.train import load_checkpoint

model = load_checkpoint(CKPT_DIR / "ch16_sft.pt", device)

# 学習データの質問（sft_step_by_step.py と同じく種 0 で 3,000 件）
rng = random.Random(0)
train_q = {make_chat_example(rng)[0][0]["content"] for _ in range(3000)}

# 評価用の 200 問（chat_accuracy と同じく種 12345）
rng = random.Random(12345)
seen, unseen = [], []
for _ in range(200):
    messages, answer = make_chat_example(rng)
    q = messages[0]["content"]
    ok = chat_reply(model, tok, q, device=device) == f"{answer}です。"
    (seen if q in train_q else unseen).append(ok)
for name, r in [("全 200 問", seen + unseen), ("学習データと同じ質問", seen),
                ("学習データにない質問", unseen)]:
    print(f"{name}: {len(r)} 問, 正答率 {sum(r) / len(r):.1%}")

# ドリル形式でたずねたときの続き（温度 0）
for prompt in ["\n12+35=", "\n300-123=", "\n7+8="]:
    reply = complete(model, tok, prompt, max_new_tokens=12, device=device)
    print(f"ドリル形式 {prompt.strip()!r} の続き: {reply!r}")
