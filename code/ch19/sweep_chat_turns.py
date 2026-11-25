"""prefix_cache.py の [3]（8 回の質問）を、回ごとの内訳に分けて記録する。

毎回、それまでの会話の全体を送り直す。キャッシュなしでは全体を、
前の回の KV をスロットに残すときは、一致しない後ろの部分だけをプリフィルする。
時間は測らず、回ごとのトークン数だけを残す（図のため）。
"""

import csv

import torch

from engine import Engine, pick
from llmbook.chat import render_chat
from prefix_cache import END, SLOT, TURNS, common_prefix, model, tok

engine, held, messages = Engine(model, n_slots=1), [], []
rows = []
with torch.no_grad():
    for k, (q, _) in enumerate(TURNS, start=1):
        messages.append({"role": "user", "content": q})
        ids = tok.encode(render_chat(messages, add_generation_prompt=True))
        n = min(common_prefix(held, ids), len(ids) - 1)  # 残っている分
        engine.length[0] = n
        nxt = pick(engine.forward(SLOT, torch.tensor([ids[n:]])))
        out = []
        while int(nxt) != END and len(out) < 5:
            out.append(int(nxt))
            nxt = pick(engine.forward(SLOT, nxt[:, None]))
        held = ids + out  # いまスロットにキーとバリューがあるトークン
        messages.append({"role": "assistant", "content": tok.decode(out)})
        rows.append((k, len(ids), n, len(ids) - n, len(out)))
        print(f"{k} 回目: 送られた列 {len(ids):3d} トークン"
              f"  残っていた分 {n:3d}  計算した分 {len(ids) - n:3d}"
              f"  応答 {len(out)} トークン（{tok.decode(out)}）")
print(f"のべ: キャッシュなし {sum(r[1] for r in rows)}"
      f"  残す {sum(r[3] for r in rows)} トークン")
with open("ch19/out/sweep_chat_turns.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["turn", "total", "kept", "computed", "reply"])
    w.writerows(rows)
