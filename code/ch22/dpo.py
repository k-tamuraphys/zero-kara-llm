"""DPO：選好の対だけから、報酬モデルも生成も使わずに方策を直接学習する。
  python ch22/dpo.py [beta] [alpha]   alpha は好まれた応答の NLL 項の重み"""
import copy
import csv
import random
import sys
import time

import torch
import torch.nn.functional as F

from common import OUT_DIR, load_sft, tok, chat_accuracy, device
from llmbook.chat import chat_reply, make_chat_example
from prefs import make_prefs, seq_logprob

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
torch.manual_seed(0)
policy = load_sft()
ref = copy.deepcopy(policy).eval()
for p in ref.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam(policy.parameters(), lr=1e-5)
train, test = make_prefs(2000, seed=1), make_prefs(200, seed=2)


def style_eval(model, n=200):
    """貪欲生成で「答えは」と言い切る割合と、数が正しい割合。"""
    rng = random.Random(12345)
    polite = correct = 0
    for _ in range(n):
        msgs, ans = make_chat_example(rng)
        reply = chat_reply(model, tok, msgs[0]["content"], device=device)
        polite += reply.startswith("答えは")
        correct += reply in (f"{ans}です。", f"答えは{ans}です。")
    return polite / n, correct / n


# [snippet: dpo-loss]
def dpo_loss(batch):
    P, W, L = zip(*batch)
    lw, ll = seq_logprob(policy, P, W), seq_logprob(policy, P, L)
    with torch.no_grad():
        rw, rl = seq_logprob(ref, P, W), seq_logprob(ref, P, L)
    h_w = beta * (lw - rw)          # 好まれた応答の「暗黙の報酬」
    h_l = beta * (ll - rl)          # 好まれなかった応答の「暗黙の報酬」
    loss = -F.logsigmoid(h_w - h_l).mean()
    n_w = torch.tensor([len(y) for y in W], device=device)
    loss = loss + alpha * (-lw / n_w).mean()  # 好まれた応答の NLL（任意）
    return loss, h_w.detach(), h_l.detach()
# [end: dpo-loss]


log, t0, B = [], time.time(), 16
for step in range(201):
    if step % 25 == 0:
        policy.eval()
        with torch.no_grad():
            _, hw, hl = dpo_loss(test)
        polite, correct = style_eval(policy)
        row = dict(step=step, pref_acc=(hw > hl).float().mean().item(),
                   chosen=hw.mean().item() / beta,
                   rejected=hl.mean().item() / beta,
                   polite=polite, correct=correct)
        log.append(row)
        print(" ".join(f"{k} {v:.3f}" if isinstance(v, float) else
                       f"{k} {v}" for k, v in row.items()), flush=True)
    if step == 200:
        break
    policy.train()
    batch = random.Random(step).sample(train, B)
    loss, _, _ = dpo_loss(batch)
    opt.zero_grad()
    loss.backward()
    opt.step()

with open(OUT_DIR / f"dpo_beta{beta}_alpha{alpha}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0]))
    w.writeheader()
    w.writerows(log)
for q in ["12+35は？", "300から123を引くといくつですか？", "7足す8は？"]:
    print(q, "->", chat_reply(policy, tok, q, device=device))
print(f"total {time.time()-t0:.0f}s")
