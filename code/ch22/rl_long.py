"""GRPO を長く回す版（22.6 節の本実験）。rl_arith.py と同じ損失を使い、
速くするために次の 3 つだけを変えている。
  1. 同じ長さのプロンプトをまとめて 1 回の generate で生成する
  2. 方策の更新は Q×G 本の右側をパディングして 1 回の forward にまとめる
  3. mu=1（1 回の生成で 1 回更新）なので π_old = π_θ。確率比は
     exp(logp − logp.detach()) とし、π_old の forward を省く
評価は学習に使わない固定の 1,000 問（種 777、貪欲）で行う。
  python ch22/rl_long.py --threads 4 --tag q32 \
      --save --out ch22_rl.pt      # 22.6 節の q32（ch22_rl.pt）
"""
import argparse
import copy
import csv
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from common import (OUT_DIR, CKPT_DIR, END, PAD, device, load_sft, prompt_ids,
                    reward, make_chat_example, save_checkpoint)
from llmbook.chat import QUESTION_TEMPLATES

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=250)
ap.add_argument("--Q", type=int, default=32)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--hard", type=float, default=0.0)  # 3 桁どうしの問題の割合
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tag", default="long")
ap.add_argument("--save", action="store_true")
ap.add_argument("--out", default="")  # 保存名（既定 ch22_rl_<tag>.pt）
args = ap.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)
rng = random.Random(0)


def make_example(rng):
    """訓練用の問題。割合 hard で 3 桁どうしの問題にする。"""
    if rng.random() >= args.hard:
        msgs, ans = make_chat_example(rng)
        return msgs[0]["content"], ans
    op = rng.choice("+-")
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    if op == "-" and a < b:
        a, b = b, a
    q = rng.choice(QUESTION_TEMPLATES[op]).format(a=a, b=b)
    return q, str(a + b if op == "+" else a - b)


@torch.no_grad()
def generate_grouped(model, prompts, n_each, temperature, max_new=8):
    """長さの同じプロンプトをまとめて生成する。返り値は prompts の順の
    応答トークン (n_each, max_new) のリスト。"""
    model.eval()
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    outs = [None] * len(prompts)
    for L, ids in by_len.items():
        idx = torch.tensor([prompts[i] for i in ids for _ in range(n_each)],
                           device=device)
        out = model.generate(idx, max_new, temperature=temperature,
                             stop_token=END)[:, L:]
        if out.size(1) < max_new:  # 全員が早く終えたら END で埋める
            out = F.pad(out, (0, max_new - out.size(1)), value=END)
        for k, i in enumerate(ids):
            outs[i] = out[k * n_each:(k + 1) * n_each]
    return outs


EVAL_RNG = random.Random(777)
EVAL = [make_chat_example(EVAL_RNG) for _ in range(1000)]
EVAL_P = [prompt_ids(m[0]["content"]) for m, _ in EVAL]


def eval1000(model):
    outs = generate_grouped(model, EVAL_P, 1, 0.0)
    hits = [reward(o[0].tolist(), a) for o, (_, a) in zip(outs, EVAL)]
    return sum(hits) / 1000


def batch_logprobs(model, seqs, P_lens, T):
    """右を埋めた (N, P+T) の列から、応答部分 T トークンの log π を返す。"""
    L = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), L), PAD, device=device)
    for k, s in enumerate(seqs):
        idx[k, :len(s)] = torch.tensor(s, device=device)
    logits, _ = model(idx[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    pos = torch.tensor(P_lens, device=device).unsqueeze(1) - 1 + \
        torch.arange(T, device=device)                      # (N, T)
    tgt = idx.gather(1, pos + 1)
    lp = logp.gather(1, pos.unsqueeze(2).expand(-1, -1, logp.size(2)))
    return lp.gather(2, tgt.unsqueeze(2)).squeeze(2)        # (N, T)


policy = load_sft()
ref = copy.deepcopy(policy).eval()
for p in ref.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

log, t0 = [], time.time()
for step in range(args.steps + 1):
    if step % args.eval_every == 0:
        acc = eval1000(policy)
        log.append({"step": step, "acc": acc})
        print(f"step {step:5d}  acc1000 {acc:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
    if step == args.steps:
        break
    exs = [make_example(rng) for _ in range(args.Q)]
    prompts = [prompt_ids(q) for q, _ in exs]
    outs = generate_grouped(policy, prompts, args.G, args.temp)
    seqs, P_lens, masks, A_all, rs, n_sig = [], [], [], [], [], 0
    for p, (q, ans), resp in zip(prompts, exs, outs):
        r = torch.tensor([reward(o.tolist(), ans) for o in resp])
        A = (r - r.mean()) / (r.std() + 1e-4)           # GRPO のアドバンテージ
        n_sig += int(r.std() > 0)
        rs.append(r.mean().item())
        is_end = (resp == END).long()
        mask = ((is_end.cumsum(1) - is_end) == 0).float()
        for o, m, a in zip(resp, mask, A):
            seqs.append(p + o.tolist()); P_lens.append(len(p))
            masks.append(m); A_all.append(a)
    mask = torch.stack(masks); A = torch.stack(A_all).unsqueeze(1)
    T = mask.size(1)
    policy.train()
    lp = batch_logprobs(policy, seqs, P_lens, T)
    with torch.no_grad():
        lp_ref = batch_logprobs(ref, seqs, P_lens, T)
    ratio = torch.exp(lp - lp.detach())  # 値は 1、勾配は ∇log π
    d = lp_ref - lp
    kl = torch.exp(d) - d - 1
    per_tok = ratio * A - args.beta * kl
    per_seq = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)
    loss = -per_seq.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    if step % args.eval_every == 0:
        klm = ((kl.detach() * mask).sum() / mask.sum()).item()
        log[-1].update(reward=sum(rs) / len(rs), signal=n_sig / args.Q, kl=klm)
        print(f"            reward {log[-1]['reward']:.3f}  signal "
              f"{log[-1]['signal']:.2f}  kl {klm:.4f}", flush=True)
    elif step < 5:
        print(f"  step {step} {time.time()-t0:.1f}s", flush=True)

keys = ["step", "acc", "reward", "signal", "kl"]
with open(OUT_DIR / f"rl_{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval="")
    w.writeheader()
    w.writerows(log)
print(f"total {time.time()-t0:.0f}s")
if args.save:
    best = max(log, key=lambda r: r["acc"])
    out = CKPT_DIR / (args.out or f"ch22_rl_{args.tag}.pt")
    save_checkpoint(policy, out,
                    method="grpo", acc1000=log[-1]["acc"], steps=args.steps,
                    lr=args.lr, Q=args.Q, G=args.G, hard=args.hard)
    print("saved", out, "best", best)
