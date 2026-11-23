"""成功軌跡の再利用（棄却サンプリング＋SFT）。各ステップで Q 問 × G 本を
ロールアウトし、グループで報酬が最大かつ正解の軌跡だけを、モデルが書いた
トークンについて SFT する。報酬は env.reward（正解 1、電卓 1 回 −c）。
  python train_raft.py --c 0.0 --name raft_c0
"""
import argparse, csv, random, time
import torch
from env import *

ap = argparse.ArgumentParser()
ap.add_argument("--c", type=float, default=0.0)
ap.add_argument("--eps", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--Q", type=int, default=32)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--minutes", type=float, default=6.0)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--name", default="raft_c0")
ap.add_argument("--warmup", type=int, default=20)  # 学習率を 0 から線形に上げるステップ数
args = ap.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
rng = random.Random(2024)
model = load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
KINDS = ["add", "add", "mul", "two"]
log = open(OUT_DIR / f"train_{args.name}.csv", "w", newline="")
w = csv.writer(log)
w.writerow(["step", "sec", "reward", "add_acc", "add_tool", "mul_acc",
            "two_acc", "kept", "loss"])
t0, step = time.time(), 0
while time.time() - t0 < args.minutes * 60:
    step += 1
    tasks = [make_task(rng, rng.choice(KINDS)) for _ in range(args.Q)]
    # [snippet: raft-step]
    group = [t for t in tasks for _ in range(args.G)]
    trajs = rollout(model, group, temperature=1.0, eps=args.eps, gen=gen)
    r = torch.tensor([reward(t, args.c) for t in trajs]).view(args.Q, args.G)
    best = (r == r.max(1, keepdim=True).values) & (r > 0)  # 組の最良の正解
    keep = [t for t, b in zip(trajs, best.view(-1)) if b]
    model.train()
    logp, m = traj_logprobs(model, keep)
    loss = -(logp * m).sum() / m.sum()        # 選んだ軌跡の SFT（マスク付き）
    for g in opt.param_groups:                  # 学習率のウォームアップ
        g["lr"] = args.lr * min(1.0, step / args.warmup)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    # [end: raft-step]
    def stat(kind, f):
        xs = [f(t) for t in trajs if t.task.kind == kind]
        return sum(xs) / max(len(xs), 1)
    corr = lambda t: float(reward(t, 0) == 1.0)
    w.writerow([step, round(time.time() - t0), round(r.mean().item(), 4),
                round(stat("add", corr), 4),
                round(stat("add", lambda t: float(t.n_calls > 0)), 4),
                round(stat("mul", corr), 4), round(stat("two", corr), 4),
                len(keep), round(loss.item(), 4)])
    log.flush()
save_checkpoint(model, CKPT_DIR / f"ch24_{args.name}.pt", c=args.c,
                steps=step, lr=args.lr, eps=args.eps, method="raft")
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
