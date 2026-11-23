"""ch23_tool_sft.pt を、軌跡単位の報酬（正解 1、電卓 1 回につき −c）と
GRPO で鍛える。  python train_agent_rl.py --c 0.0 --name c0
"""
import argparse, copy, csv, random, re, time
import torch
from env import *

ap = argparse.ArgumentParser()
ap.add_argument("--c", type=float, default=0.0)      # 電卓 1 回のコスト
ap.add_argument("--eps", type=float, default=0.2)    # 最初の役割の強制探索
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--Q", type=int, default=16)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--minutes", type=float, default=7.0)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--name", default="c0")
ap.add_argument("--init", default="ch23_tool_sft")  # 出発点
ap.add_argument("--step0", type=int, default=0)     # 続きのときのステップの始まり
ap.add_argument("--warmup", type=int, default=20)  # 学習率を 0 から線形に上げるステップ数
args = ap.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
rng = random.Random(2024)

model = load_checkpoint(CKPT_DIR / f"{args.init}.pt")
ref = load_checkpoint(CKPT_DIR / f"{args.init}.pt").eval()  # 参照 π_ref
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
KINDS = ["add", "add", "mul", "two"]                  # 加減算を半分

log = open(OUT_DIR / f"train_{args.name}.csv", "a" if args.step0 else "w", newline="")
w = csv.writer(log)
if not args.step0: w.writerow(["step", "sec", "reward", "add_acc", "add_tool", "mul_acc",
            "two_acc", "informative", "kl"])
t0, step = time.time(), args.step0
rng = random.Random(2024 + args.step0)
while time.time() - t0 < args.minutes * 60:
    step += 1
    tasks = [make_task(rng, rng.choice(KINDS)) for _ in range(args.Q)]
    # [snippet: grpo-step]
    group = [t for t in tasks for _ in range(args.G)]        # 1 問に G 本
    trajs = rollout(model, group, temperature=1.0, eps=args.eps, gen=gen)
    r = torch.tensor([reward(t, args.c) for t in trajs]).view(args.Q, args.G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    adv = adv.view(-1)                                       # 軌跡ごとに 1 つ
    model.train()
    logp, m = traj_logprobs(model, trajs)                    # (N, L-1)
    with torch.no_grad():
        logp_ref, _ = traj_logprobs(ref, trajs)
    d = logp_ref - logp
    kl = torch.exp(d) - d - 1                                # KL の推定 k3
    # モデルが書いたトークン（m=1）だけ、軌跡のアドバンテージで押す
    loss = ((-adv[:, None] * logp + args.beta * kl) * m).sum() / m.sum()
    for g in opt.param_groups:                  # 学習率のウォームアップ
        g["lr"] = args.lr * min(1.0, step / args.warmup)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    # [end: grpo-step]
    def stat(kind, f):
        xs = [f(t) for t in trajs if t.task.kind == kind]
        return sum(xs) / max(len(xs), 1)
    corr = lambda t: float(reward(t, 0) == 1.0)
    w.writerow([step, round(time.time() - t0), round(r.mean().item(), 4),
                round(stat("add", corr), 4),
                round(stat("add", lambda t: float(t.n_calls > 0)), 4),
                round(stat("mul", corr), 4), round(stat("two", corr), 4),
                round((r.std(1) > 0).float().mean().item(), 4),
                round(((kl * m).sum() / m.sum()).item(), 5)])
    log.flush()
    if step % 10 == 0:
        print(f"step {step} {time.time()-t0:.0f}s reward {r.mean():.3f}"
              f" add_acc {stat('add', corr):.2f}"
              f" add_tool {stat('add', lambda t: float(t.n_calls>0)):.2f}",
              flush=True)

save_checkpoint(model, CKPT_DIR / f"ch24_{args.name}.pt", c=args.c,
                steps=step, lr=args.lr, eps=args.eps)
res, trs = evaluate(model, eval_sets())
print(f"学習 {step} ステップ（{time.time()-t0:.0f} 秒）, c={args.c}")
for k, v in res.items():
    print(f"  {k}: 正答率 {v['acc']:.3f}  電卓使用率 {v['tool']:.3f}"
          f"  平均呼び出し {v['calls']:.2f}")
with open(OUT_DIR / f"eval_{args.name}.csv", "w", newline="") as f:
    ww = csv.writer(f)
    ww.writerow(["kind", "question", "answer", "calls", "correct", "traj"])
    for k, ts in trs.items():
        for t in ts:
            ww.writerow([k, t.task.question, t.task.answer, t.n_calls,
                         int(reward(t, 0) == 1.0), "".join(t.texts)])
print(f"[合計 {time.time()-t0:.0f} 秒]")
