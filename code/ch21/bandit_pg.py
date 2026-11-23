"""3 択バンディットで、方策勾配の推定値を厳密な勾配と比べる。

計算アシスタントが「12+35は？」に 3 通りの答え方を持つとし、
正解「47です。」だけに報酬 1 を与える。
"""
import sys
import torch

torch.manual_seed(0)
OUT = "ch21/out"
sys.stdout = open(f"{OUT}/bandit_exact.txt", "w")   # 出力を部分ごとに分ける

# [snippet: bandit-setup]
answers = ["47です。", "46です。", "わかりません。"]
r = torch.tensor([1.0, 0.0, 0.0])          # 各行動の報酬 r(a)
theta = torch.zeros(3, requires_grad=True)  # ロジット（方策のパラメータ）

pi = torch.softmax(theta, dim=0)            # 方策 π_θ(a)
J = (pi * r).sum()                          # 期待報酬 J(θ)=Σ π(a) r(a)
J.backward()                                # 期待値を全部書けるので微分できる
print("π      =", pi.detach().numpy().round(4))
print("J      =", round(J.item(), 4))
print("∇J(厳密) =", theta.grad.numpy().round(4))
# [end: bandit-setup]


# [snippet: score-estimate]
def score_estimates(theta, r, n, b=0.0):
    """n 個のサンプルそれぞれの勾配推定 (r(a)-b)∇log π(a) を返す。"""
    pi = torch.softmax(theta.detach(), dim=0)
    a = torch.multinomial(pi, n, replacement=True)   # 行動をサンプル (n,)
    onehot = torch.nn.functional.one_hot(a, 3).float()
    score = onehot - pi                              # ∇_θ log π(a) (n, 3)
    return (r[a] - b)[:, None] * score               # (n, 3)
# [end: score-estimate]


sys.stdout = open(f"{OUT}/bandit_estimate.txt", "w")
# [snippet: score-check]
for n in [10, 100, 1000, 100000]:
    g = score_estimates(theta, r, n).mean(dim=0)
    print(f"n={n:>6}: 推定 ∇J =", g.numpy().round(4))
# [end: score-check]

# 推定値（成分 1）が n とともに厳密値に近づく様子（図用）
torch.manual_seed(1)
g_all = score_estimates(theta, r, 10000)
running = g_all.cumsum(0) / torch.arange(1, 10001)[:, None]
with open(f"{OUT}/bandit_running.csv", "w") as f:
    f.write("n,g1,g2,g3\n")
    for n in sorted({int(round(10 ** (k / 20))) for k in range(0, 81)}):
        g1, g2, g3 = running[n - 1].tolist()
        f.write(f"{n},{g1:.5f},{g2:.5f},{g3:.5f}\n")

sys.stdout = open(f"{OUT}/bandit_variance.txt", "w")
# [snippet: variance]
J0 = (torch.softmax(theta.detach(), 0) * r).sum()
for shift in [0.0, 10.0]:
    for b_name, b in [("b=0", 0.0), ("b=J", J0 + shift)]:
        torch.manual_seed(2)
        g = score_estimates(theta, r + shift, 200000, b)
        print(f"報酬+{shift:<4} {b_name}: 平均",
              g.mean(0).numpy().round(3),
              f" 分散 {g.var(0).sum().item():8.4f}")
# [end: variance]


# [snippet: bandit-train]
def train(shift, use_baseline, steps=100, batch=4, lr=0.5, seed=0):
    torch.manual_seed(seed)
    th, hist = torch.zeros(3), []
    b = (r + shift)[torch.randint(0, 3, (batch,))].mean().item()  # 初期値
    for _ in range(steps):
        pi = torch.softmax(th, 0)
        a = torch.multinomial(pi, batch, replacement=True)
        R = (r + shift)[a]                           # 観測した報酬 (batch,)
        score = torch.nn.functional.one_hot(a, 3).float() - pi
        adv = R - b if use_baseline else R
        th = th + lr * (adv[:, None] * score).mean(0)   # 勾配「上昇」
        b = 0.8 * b + 0.2 * R.mean().item()          # 過去の報酬の移動平均
        hist.append(torch.softmax(th, 0)[0].item())  # 正解の確率
    return hist
# [end: bandit-train]


runs = {name: [train(s, ub, seed=k) for k in range(20)]
        for name, s, ub in [("r", 0.0, False), ("r10", 10.0, False),
                            ("r10b", 10.0, True)]}
with open(f"{OUT}/bandit_train.csv", "w") as f:
    f.write("step," + ",".join(f"{k}_mean,{k}_lo,{k}_hi" for k in runs) + "\n")
    for t in range(100):
        row = [str(t + 1)]
        for k, rs in runs.items():
            v = sorted(h[t] for h in rs)
            row += [f"{sum(v)/len(v):.4f}", f"{v[2]:.4f}", f"{v[-3]:.4f}"]
        f.write(",".join(row) + "\n")
sys.stdout = open(f"{OUT}/bandit_train.txt", "w")
print("100 ステップ後の正解の確率（20 回の平均）")
for k, rs in runs.items():
    print(f"{k:>5}: {sum(h[-1] for h in rs)/20:.3f}")
