"""多腕バンディットと ε-貪欲法: 探索と活用のつり合いを見る。

実行: cd code && python ch20/bandit.py
"""
import sys
import numpy as np

PROBS = np.array([0.2, 0.8, 0.5, 0.3, 0.6])   # 5 台それぞれの当たる確率
BEST = int(np.argmax(PROBS))                  # いちばん良い台（1 番）


# [snippet: bandit]
# [snippet: eps-greedy]
def epsilon_greedy(q, eps, rng):
    """確率 ε でランダムに選び（探索）、残りは最大の行動を選ぶ（活用）。"""
    if rng.random() < eps:
        return int(rng.integers(len(q)))
    best = np.flatnonzero(q == q.max())        # 最大が複数あれば
    return int(rng.choice(best))               # その中からランダムに
# [end: eps-greedy]


def run_bandit(eps, n_steps, rng):
    """1 人のプレイヤーが n_steps 回引く。各回の報酬と選んだ台を返す。"""
    Q = np.zeros(len(PROBS))                   # 各台の価値の見積もり
    N = np.zeros(len(PROBS))                   # 各台を引いた回数
    rewards, picks = np.zeros(n_steps), np.zeros(n_steps, dtype=int)
    for t in range(n_steps):
        a = epsilon_greedy(Q, eps, rng)
        r = float(rng.random() < PROBS[a])     # 当たれば 1、外れれば 0
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]              # これまでの平均を更新
        rewards[t], picks[t] = r, a
    return rewards, picks
# [end: bandit]


if __name__ == "__main__":
    OUT = "ch20/out"
    n_runs, n_steps = 2000, 500
    eps_list = [0.0, 0.1, 0.5]
    avg_r = np.zeros((len(eps_list), n_steps))
    best_rate = np.zeros((len(eps_list), n_steps))
    sys.stdout = open(f"{OUT}/bandit.txt", "w")
    print("  ε    平均報酬   最良の台を選ぶ割合（最後の 100 回）")
    for i, eps in enumerate(eps_list):
        rng = np.random.default_rng(0)
        for run in range(n_runs):
            rewards, picks = run_bandit(eps, n_steps, rng)
            avg_r[i] += rewards / n_runs
            best_rate[i] += (picks == BEST) / n_runs
        print(f"{eps:>4}     {avg_r[i].mean():.3f}       "
              f"{best_rate[i, -100:].mean():.3f}")
    with open(f"{OUT}/bandit.csv", "w") as f:
        f.write("step,r0,r01,r05,b0,b01,b05\n")
        for t in range(n_steps):
            row = list(avg_r[:, t]) + list(best_rate[:, t])
            f.write(f"{t + 1}," + ",".join(f"{x:.4f}" for x in row) + "\n")
