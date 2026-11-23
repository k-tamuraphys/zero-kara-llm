"""REINFORCE（とその改良版）で倒立振子を学習する。

method="total"   : すべての時刻を軌跡全体の収益 R(τ) で重み付け
method="rtg"     : 時刻 t 以降の収益 G_t（reward-to-go）で重み付け
method="baseline": G_t から価値ネット V_φ(s_t) を引いたアドバンテージで重み付け
"""
import sys
import time
import torch
import torch.nn as nn
from cartpole import CartPole

torch.set_num_threads(2)
OUT = "ch21/out"


# [snippet: policy-net]
class PolicyNet(nn.Module):
    """状態 s (4 次元) → 2 つの行動のロジット → softmax 方策。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2))

    def dist(self, s):                       # s: (..., 4)
        return torch.distributions.Categorical(logits=self.net(s))
# [end: policy-net]


def value_net(hidden=64):
    return nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, 1))


# [snippet: rollout]
@torch.no_grad()
def rollout(policy, n):
    """n エピソードを同時に最後まで走らせ、(T, n) 形の記録を返す。"""
    env = CartPole(n)
    s = env.reset()
    alive = torch.ones(n, dtype=torch.bool)   # まだ倒れていないか
    S, A, R, M = [], [], [], []
    for t in range(env.max_steps):
        a = policy.dist(s).sample()           # a_t ~ π_θ(·|s_t)
        s_next, r, done = env.step(a)
        S.append(s); A.append(a); R.append(r * alive); M.append(alive)
        alive = alive & ~done                 # 倒れたらそこで終わり
        s = s_next
        if not alive.any():
            break
    return (torch.stack(S), torch.stack(A),   # (T, n, 4), (T, n)
            torch.stack(R), torch.stack(M).float())
# [end: rollout]


# [snippet: reward-to-go]
def reward_to_go(R, gamma):
    """G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... を後ろから計算する。"""
    G = torch.zeros_like(R)
    running = torch.zeros(R.shape[1])
    for t in reversed(range(R.shape[0])):
        running = R[t] + gamma * running      # G_t = r_t + γ G_{t+1}
        G[t] = running
    return G
# [end: reward-to-go]


# [snippet: reinforce-update]
def train(method, seed, iters=80, n_ep=16, lr=0.01, gamma=0.99,
          log_every=0):
    torch.manual_seed(seed)
    policy, V = PolicyNet(), value_net()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_v = torch.optim.Adam(V.parameters(), lr=0.01)
    history = []
    for it in range(iters):
        S, A, R, M = rollout(policy, n_ep)
        G = reward_to_go(R, gamma)                    # (T, n)
        if method == "total":
            w = G[0].expand_as(G)                     # 全時刻に R(τ)
        elif method == "rtg":
            w = G                                     # G_t
        else:
            w = G - V(S).squeeze(-1).detach()         # G_t − V_φ(s_t)
        logp = policy.dist(S).log_prob(A)             # log π_θ(a_t|s_t)
        loss = -(w * logp * M).sum() / n_ep           # 符号を反転して最小化
        opt.zero_grad(); loss.backward(); opt.step()
        if method == "baseline":                      # V_φ を G_t に回帰
            for _ in range(5):
                v_loss = (((V(S).squeeze(-1) - G) ** 2) * M).sum() / M.sum()
                opt_v.zero_grad(); v_loss.backward(); opt_v.step()
        history.append(M.sum(0).mean().item())        # 平均エピソード長
        if log_every and (it + 1) % log_every == 0:
            print(f"iter {it+1:3d}  平均エピソード長 {history[-1]:6.1f}")
    return history, policy, V
# [end: reinforce-update]


def summarize(runs):
    """種ごとの曲線 → 各反復の平均・最小・最大。"""
    T = len(runs[0])
    cols = list(zip(*runs))
    return [(sum(c) / len(c), min(c), max(c)) for c in cols][:T]


if __name__ == "__main__":
    t0 = time.time()
    seeds = range(5)
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        train("baseline", seed=0, log_every=10)
        print(f"所要時間 {time.time()-t0:.0f} 秒")
        sys.exit()
    results = {}
    for method in ["total", "rtg", "baseline"]:
        runs = [train(method, s)[0] for s in seeds]
        results[method] = summarize(runs)
        last = [sum(r[-10:]) / 10 for r in runs]
        print(f"{method:>8}: 最後の 10 反復の平均エピソード長（種ごと）",
              " ".join(f"{x:5.1f}" for x in last),
              f" 平均 {sum(last)/len(last):5.1f}")
    with open(f"{OUT}/reinforce_curves.csv", "w") as f:
        f.write("episodes," + ",".join(
            f"{m}_mean,{m}_lo,{m}_hi" for m in results) + "\n")
        for i in range(len(results["total"])):
            row = [str((i + 1) * 16)]
            for m in results:
                row += [f"{v:.1f}" for v in results[m][i]]
            f.write(",".join(row) + "\n")
    print(f"所要時間 {time.time()-t0:.0f} 秒")
