"""GRPO に PPO の「同じ生成を複数回使う」を持ち込む版（22.6 節の主実験）。
rl_long.py との違いは次の 4 つ。
  1. 生成の直後に log π_old を計算して保存する
  2. 同じ生成で K エポック更新し、確率比を ε でクリップする（クリップが働く）
  3. 報酬が全部同じグループ（アドバンテージが 0）は更新から除く
  4. 生成時の方策との近似 KL が kl_stop を超えたら、残りのエポックを打ち切る
最良モデルは検証用 500 問（種 555）で選び、報告はテスト用 1,000 問（種 777）。
  python ch22/rl_ppo.py --threads 2 --tag ppo_a --save
"""
import argparse
import copy
import csv
import random
import time

import torch

from common import OUT_DIR, CKPT_DIR, device, load_sft, prompt_ids, reward, \
    make_chat_example, save_checkpoint, END
from rl_lib import (make_hard_example, generate_grouped, batch_logprobs,
                         eval_acc, EVAL_TEST, EVAL_VAL)

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=150)  # 生成の回数
ap.add_argument("--Q", type=int, default=64)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--K", type=int, default=4)        # 1 回の生成を使う回数
ap.add_argument("--eps", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=20)  # 更新回数で数える
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--kl_stop", type=float, default=0.04)
ap.add_argument("--hard", type=float, default=0.5)
ap.add_argument("--std", type=int, default=1)  # 0: r − 平均（Dr. GRPO 流）
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=10)  # 生成の回数で数える
ap.add_argument("--max_time", type=float, default=1e9)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tag", default="ppo")
ap.add_argument("--save", action="store_true")
ap.add_argument("--out", default="")
args = ap.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)
rng = random.Random(0)


def make_example(rng):
    if rng.random() < args.hard:
        return make_hard_example(rng)
    msgs, ans = make_chat_example(rng)
    return msgs[0]["content"], ans


policy = load_sft()
ref = copy.deepcopy(policy).eval()
for p in ref.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

# [snippet: ppo-rollout]
def collect(policy, exs):
    """Q 問 × G 本を生成し、信号のあるグループだけを返す。
    log π_old は生成直後の方策で 1 回だけ計算して保存しておく。"""
    prompts = [prompt_ids(q) for q, _ in exs]
    outs = generate_grouped(policy, prompts, args.G, args.temp)
    seqs, P_lens, masks, A_all, rs = [], [], [], [], []
    for p, (q, ans), resp in zip(prompts, exs, outs):
        r = torch.tensor([reward(o.tolist(), ans) for o in resp])
        rs.append(r.mean().item())
        if r.std() == 0:            # 全問正解・全問不正解：A = 0 なので捨てる
            continue
        A = r - r.mean()
        if args.std:
            A = A / (r.std() + 1e-4)
        is_end = (resp == END).long()
        mask = ((is_end.cumsum(1) - is_end) == 0).float()
        for o, m, a in zip(resp, mask, A):
            seqs.append(p + o.tolist()); P_lens.append(len(p))
            masks.append(m); A_all.append(a)
    if not seqs:
        return None, rs
    mask = torch.stack(masks); A = torch.stack(A_all).unsqueeze(1)
    with torch.no_grad():
        lp_old = batch_logprobs(policy, seqs, P_lens, mask.size(1))
        lp_ref = batch_logprobs(ref, seqs, P_lens, mask.size(1))
    return (seqs, P_lens, mask, A, lp_old, lp_ref), rs
# [end: ppo-rollout]


log, t0, n_upd, best = [], time.time(), 0, (-1.0, None, 0)
for it in range(args.iters + 1):
    if it % args.eval_every == 0 or it == args.iters or \
            time.time() - t0 > args.max_time:
        va, te = eval_acc(policy, EVAL_VAL), eval_acc(policy, EVAL_TEST)
        log.append({"iter": it, "updates": n_upd, "val": va, "test": te})
        if va > best[0]:
            best = (va, copy.deepcopy(policy.state_dict()), it)
        print(f"iter {it:4d} upd {n_upd:4d}  val {va:.3f}  test {te:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if it == args.iters or time.time() - t0 > args.max_time:
            break
    batch, rs = collect(policy, [make_example(rng) for _ in range(args.Q)])
    if batch is None:
        continue
    seqs, P_lens, mask, A, lp_old, lp_ref = batch
    # [snippet: ppo-epochs]
    policy.eval()  # ドロップアウトは 0。生成時と同じ関数で比を測る
    for epoch in range(args.K):
        for g in opt.param_groups:  # 線形ウォームアップ
            g["lr"] = args.lr * min(1.0, (n_upd + 1) / args.warmup)
        lp = batch_logprobs(policy, seqs, P_lens, mask.size(1))
        ratio = torch.exp(lp - lp_old)              # 2 エポック目からは ≠ 1
        clipped = ratio.clamp(1 - args.eps, 1 + args.eps)
        surr = torch.min(ratio * A, clipped * A)    # PPO のクリップ
        d = lp_ref - lp
        kl = torch.exp(d) - d - 1                   # β = 0 なら KL 罰則なし
        per_tok = surr - args.beta * kl if args.beta > 0 else surr
        per_seq = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = -per_seq.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        n_upd += 1
        with torch.no_grad():  # 生成時の方策との近似 KL（更新前の値）
            dd = lp.detach() - lp_old
            approx_kl = ((torch.exp(dd) - 1 - dd) * mask).sum() / mask.sum()
            frac = (((ratio - 1).abs() > args.eps).float() * mask).sum() \
                / mask.sum()
        if approx_kl > args.kl_stop:    # 動きすぎたら、この生成はここまで
            break
    # [end: ppo-epochs]
    klr = ((kl.detach() * mask).sum() / mask.sum()).item()
    log.append({"iter": it, "updates": n_upd, "reward": sum(rs) / len(rs),
                "kept": len(seqs) // args.G / args.Q, "epochs": epoch + 1,
                "clipfrac": frac.item(), "approx_kl": approx_kl.item(),
                "kl_ref": klr})
    if it < 3 or it % 5 == 0:
        print(f"  it {it} reward {log[-1]['reward']:.3f} kept "
              f"{log[-1]['kept']:.2f} ep {epoch+1} clip {frac:.3f} "
              f"akl {approx_kl:.4f} klref {klr:.4f} ({time.time()-t0:.0f}s)",
              flush=True)

keys = ["iter", "updates", "val", "test", "reward", "kept", "epochs",
        "clipfrac", "approx_kl", "kl_ref"]
with open(OUT_DIR / f"rl_{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval="")
    w.writeheader()
    w.writerows(log)
print(f"total {time.time()-t0:.0f}s  updates {n_upd}")
policy.load_state_dict(best[1])
te = eval_acc(policy, EVAL_TEST)
print(f"best val {best[0]:.3f} at iter {best[2]}  -> test {te:.3f}")
if args.save:
    out = CKPT_DIR / (args.out or f"ch22_rl_{args.tag}.pt")
    save_checkpoint(policy, out, method="grpo+ppo-epochs", test1000=te,
                    val500=best[0], iter=best[2], lr=args.lr, Q=args.Q,
                    G=args.G, K=args.K, eps=args.eps, hard=args.hard,
                    std=args.std)
    print("saved", out)
