"""表は「one-hot ベクトルを入れる線形層」と同じ: Q 学習を PyTorch で書く。

実行: cd code && python ch20/q_net.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "ch20")
from gridworld import N_STATES, GridWorld
from policy_eval import GAMMA
from bandit import epsilon_greedy

torch.set_num_threads(1)
torch.manual_seed(0)
ALPHA, EPS = 0.1, 0.1

# [snippet: q-net]
net = nn.Linear(N_STATES, 4, bias=False)   # 重みは 4 × 12 個 = Q の表
nn.init.zeros_(net.weight)                 # 表と同じく 0 から始める
opt = torch.optim.SGD(net.parameters(), lr=ALPHA)
onehot = torch.eye(N_STATES)               # onehot[s]: 状態 s を表す入力


def net_update(s, a, r, s2):
    """経験 (s, a, r, s') 1 個で、ネットワークを 1 回だけ更新する。"""
    with torch.no_grad():                  # 目標は定数として扱う
        target = r + GAMMA * net(onehot[s2]).max()
    loss = 0.5 * (target - net(onehot[s])[a]) ** 2
    opt.zero_grad()
    loss.backward()
    opt.step()
# [end: q-net]


if __name__ == "__main__":
    sys.stdout = open("ch20/out/q_net.txt", "w")
    Q = np.zeros((N_STATES, 4))            # 比べる相手: 表形式の Q 学習
    env, rng = GridWorld(seed=0), np.random.default_rng(20000)
    n_updates = 0
    for ep in range(300):
        s, done = env.reset(), False
        while not done:
            a = epsilon_greedy(Q[s], EPS, rng)
            s2, r, done = env.step(a)
            Q[s, a] += ALPHA * (r + GAMMA * Q[s2].max() - Q[s, a])
            net_update(s, a, r, s2)        # 同じ経験をネットワークにも
            s, n_updates = s2, n_updates + 1
    W = net.weight.detach().numpy().T      # 12 × 4 に並べ直す
    print(f"更新 {n_updates} 回のあと、表とネットワークの差の最大値: "
          f"{np.abs(W - Q).max():.1e}")
    print("表の Q[8]         =", np.round(Q[8], 4))
    print("ネットワークの出力 =", np.round(W[8], 4))
