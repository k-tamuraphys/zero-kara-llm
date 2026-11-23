"""計算アシスタント sft.pt を、正誤の報酬だけで強化学習する。
  python ch22/rl_arith.py grpo        # GRPO（ch22_rl_grpo80.pt を保存）
  python ch22/rl_arith.py reinforce   # ベースラインなしの REINFORCE
  python ch22/rl_arith.py baseline    # グループ平均を引く REINFORCE
  python ch22/rl_arith.py grpo --lr 3e-4 --tag highlr   # 学習率が高すぎる例
"""
import argparse
import copy
import csv
import random
import time

import torch

from common import (OUT_DIR, CKPT_DIR, load_sft, prompt_ids, rollout,
                    token_logprobs, reward, accuracy, make_chat_example,
                    save_checkpoint)

ap = argparse.ArgumentParser()
ap.add_argument("method", choices=["reinforce", "baseline", "grpo"])
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--Q", type=int, default=16)     # 1 ステップの問題数
ap.add_argument("--G", type=int, default=8)      # 1 問あたりの回答数
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--beta", type=float, default=0.04)  # KL 正則化の強さ
ap.add_argument("--eps", type=float, default=0.2)    # クリップ幅
ap.add_argument("--mu", type=int, default=1)     # 1 回の生成で何回更新するか
ap.add_argument("--eval_every", type=int, default=10)
ap.add_argument("--tag", default="")
ap.add_argument("--save", action="store_true")
args = ap.parse_args()
torch.manual_seed(0)
rng = random.Random(0)  # 訓練用の問題（評価は種 12345 の別の 200 問）

policy = load_sft()
ref = copy.deepcopy(policy).eval()  # 参照モデル π_ref（凍結した SFT）
for p in ref.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
mu = args.mu if args.method == "grpo" else 1


# [snippet: advantage]
def advantages(r: torch.Tensor, method: str) -> torch.Tensor:
    """r (G,) はグループ（同じ問題への G 本の回答）の報酬。"""
    if method == "reinforce":           # ベースラインなし：A = r
        return r
    if method == "baseline":            # グループ平均を引く：A = r − mean
        return r - r.mean()
    return (r - r.mean()) / (r.std() + 1e-4)  # GRPO：平均を引き標準偏差で割る
# [end: advantage]


# [snippet: grpo-loss]
def grpo_loss(logp, logp_old, logp_ref, A, mask, method, eps, beta):
    """logp* と mask は (G, T')、A は (G,)。GRPO の目的の符号を反転。"""
    A = A.unsqueeze(1)                               # (G, 1) 全トークンで共有
    if method == "grpo":
        ratio = torch.exp(logp - logp_old)           # 確率比 ρ
        surr = torch.min(ratio * A,
                         torch.clamp(ratio, 1 - eps, 1 + eps) * A)
        d = logp_ref - logp                          # log(π_ref / π_θ)
        kl = torch.exp(d) - d - 1                    # KL の推定（0 以上）
        per_tok = surr - beta * kl
    else:                                            # 素朴な方策勾配
        per_tok = logp * A
    n_tok = mask.sum(dim=1).clamp(min=1)             # 各応答の長さ |o_i|
    per_seq = (per_tok * mask).sum(dim=1) / n_tok    # 1/|o_i| Σ_t
    return -per_seq.mean()                           # 1/G Σ_i、最大化→最小化
# [end: grpo-loss]


log = []
t0 = time.time()
for step in range(args.steps + 1):
    if step % args.eval_every == 0:
        acc = accuracy(policy)
        print(f"step {step:4d}  acc {acc:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
        log.append({"step": step, "acc": acc})
    if step == args.steps:
        break
    # [snippet: train-step]
    batch, rewards, n_signal = [], [], 0
    for _ in range(args.Q):              # 1. 問題を選び G 本ずつ生成する
        msgs, ans = make_chat_example(rng)
        p = prompt_ids(msgs[0]["content"])
        resp, mask = rollout(policy, p, args.G)
        r = torch.tensor([reward(o.tolist(), ans) for o in resp])
        A = advantages(r, args.method)   # 2. 報酬からアドバンテージ
        n_signal += int(A.abs().sum() > 0)
        with torch.no_grad():            # 3. 古い方策と参照モデルの log π
            lp_old = token_logprobs(policy, p, resp)
            lp_ref = token_logprobs(ref, p, resp)
        batch.append((p, resp, mask, A, lp_old, lp_ref))
        rewards.append(r.mean().item())
    policy.train()
    for _ in range(mu):                  # 4. 同じ生成で mu 回更新する
        opt.zero_grad()
        kls = []
        for p, resp, mask, A, lp_old, lp_ref in batch:
            lp = token_logprobs(policy, p, resp)
            loss = grpo_loss(lp, lp_old, lp_ref, A, mask, args.method,
                             args.eps, args.beta) / args.Q
            loss.backward()
            d = (lp_ref - lp).detach()
            kls.append((((d.exp() - d - 1) * mask).sum() / mask.sum()).item())
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    # [end: train-step]
    log[-1].update(reward=sum(rewards) / len(rewards),
                   signal=n_signal / args.Q, kl=sum(kls) / len(kls)) \
        if step % args.eval_every == 0 else None
    if step % args.eval_every == 0:
        print(f"           reward {log[-1]['reward']:.3f}  "
              f"signal {log[-1]['signal']:.2f}  kl {log[-1]['kl']:.4f}")

name = args.method + (f"_{args.tag}" if args.tag else "")
keys = ["step", "acc", "reward", "signal", "kl"]
with open(OUT_DIR / f"rl_{name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval="")
    w.writeheader()
    w.writerows(log)
print(f"total {time.time()-t0:.0f}s")
if args.save:
    save_checkpoint(policy, CKPT_DIR / "ch22_rl_grpo80.pt",
                    method=args.method, acc=log[-1]["acc"], steps=args.steps)
    print("saved", CKPT_DIR / "ch22_rl_grpo80.pt")
