"""sft.pt と ch22_rl.pt を、200 問より大きな 1,000 問で比べる。
貪欲生成の正答率と、温度 1 でサンプリングしたときの正答率の両方を測る。"""
import random
import re
import sys

import torch

from common import (CKPT_DIR, device, load_sft, prompt_ids, reward, rollout,
                    make_chat_example)
from llmbook.train import load_checkpoint

torch.set_num_threads(2)
torch.manual_seed(0)


def measure(model, n=1000, seed=777):
    rng = random.Random(seed)
    greedy = sampled = 0.0
    hard = [0, 0]  # 3 桁どうしの問題の [正解数, 問題数]
    for _ in range(n):
        msgs, ans = make_chat_example(rng)
        p = prompt_ids(msgs[0]["content"])
        out = model.generate(torch.tensor([p], device=device), 8,
                             temperature=0.0, stop_token=None)
        g = reward(out[0, len(p):].tolist(), ans)
        greedy += g
        a, b = re.findall(r"\d+", msgs[0]["content"])[:2]
        if min(int(a), int(b)) >= 100:
            hard[0] += g
            hard[1] += 1
        resp, _ = rollout(model, p, 1, temperature=1.0)
        sampled += reward(resp[0].tolist(), ans)
    return greedy / n, sampled / n, hard


for name in sys.argv[1:] or ["sft.pt", "ch22_rl.pt"]:
    m = load_sft() if name == "sft.pt" else load_checkpoint(
        CKPT_DIR / name, device)
    m.eval()
    with torch.no_grad():
        g, s, h = measure(m)
    se = (g * (1 - g) / 1000) ** 0.5
    print(f"{name:11s} 貪欲 {g:.3f}（標準誤差 {se:.3f}）  温度1 {s:.3f}"
          f"  3桁どうし {h[0] / h[1]:.3f}（{h[1]}問）")
