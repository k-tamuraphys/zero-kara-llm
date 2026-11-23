"""方策勾配の 4 種類の推定量について、平均とばらつきを実測する。

方策を固定し、エピソード 1 本ごとの勾配推定ベクトルを多数集めて比べる。
"""
import torch
from reinforce import PolicyNet, value_net, rollout, reward_to_go, train

torch.set_num_threads(2)
OUT = "ch21/out"
gamma = 1.0                       # 割引なし: 4 つとも同じ勾配の不偏推定

# 少しだけ学習した方策（平均エピソード長が数十〜百程度）を固定して使う
_, policy, _ = train("baseline", seed=0, iters=8)
torch.manual_seed(123)

# ベースラインを作るための「別の」エピソード（推定に使うものとは独立）
S0, A0, R0, M0 = rollout(policy, 400)
G0 = reward_to_go(R0, gamma)
b_t = (G0 * M0).sum(1) / M0.sum(1).clamp(min=1)     # 時刻ごとの平均収益
V = value_net()
opt = torch.optim.Adam(V.parameters(), lr=0.01)
for _ in range(300):                                 # V_φ(s) ≈ G_t の回帰
    loss = (((V(S0).squeeze(-1) - G0) ** 2) * M0).sum() / M0.sum()
    opt.zero_grad(); loss.backward(); opt.step()

# [snippet: per-episode-grad]
S, A, R, M = rollout(policy, 1000)
G = reward_to_go(R, gamma)
T = S.shape[0]
weights = {
    "R(τ)":        G[0].expand_as(G),                # 軌跡全体の収益
    "G_t":         G,                                # reward-to-go
    "G_t - b_t":   G - b_t[:T, None],                # 時刻ごとの平均を引く
    "G_t - V(s_t)": G - V(S).squeeze(-1).detach(),   # 価値ネットを引く
}
params = list(policy.parameters())
grads = {k: [] for k in weights}
for i in range(S.shape[1]):                          # エピソード 1 本ずつ
    logp = policy.dist(S[:, i]).log_prob(A[:, i]) * M[:, i]    # (T,)
    for k, w in weights.items():
        g = torch.autograd.grad((w[:, i] * logp).sum(), params,
                                retain_graph=True)
        grads[k].append(torch.cat([x.flatten() for x in g]))
# [end: per-episode-grad]

# [snippet: variance-report]
ref = torch.stack(grads["G_t - V(s_t)"]).mean(0)
print(f"エピソード長の平均 {M.sum(0).mean():.1f}、パラメータ数 {ref.numel()}")
print("推定量          平均の長さ  基準との cos   分散の合計      比")
base_var = None
rows = []
for k, gs in grads.items():
    gs = torch.stack(gs)                             # (1000, パラメータ数)
    mean, var = gs.mean(0), gs.var(0).sum().item()
    cos = torch.cosine_similarity(mean, ref, dim=0).item()
    base_var = base_var or var
    rows.append((k, var / base_var))
    print(f"{k:<14}{mean.norm():8.2f}{cos:12.3f}{var:12.0f}"
          f"{var / base_var:8.3f}")
# [end: variance-report]
with open(f"{OUT}/grad_variance.csv", "w") as f:
    f.write("idx,ratio\n")                   # 図用（idx は上の表の行番号）
    for i, (k, v) in enumerate(rows):
        f.write(f"{i + 1},{v:.4f}\n")
