"""Q 学習: 遷移確率を知らずに、経験だけから最適な行動価値 Q* を学ぶ。

実行: cd code && python ch20/q_learning.py
"""
import sys
import numpy as np
sys.path.insert(0, "ch20")
from gridworld import N_STATES, STATES, ACTIONS, START, GridWorld
from gridworld import build_model
from policy_eval import GAMMA, policy_evaluation
from bandit import epsilon_greedy


# [snippet: q-learning]
def q_learning(env, n_episodes, alpha, gamma, eps, rng, on_episode=None):
    """表形式の Q 学習。Q[s, a] を 0 から始めて、1 歩ごとに書き換える。"""
    Q = np.zeros((N_STATES, 4))
    for ep in range(n_episodes):
        s, done = env.reset(), False
        while not done:
            a = epsilon_greedy(Q[s], eps, rng)     # ε-貪欲で行動を選ぶ
            s2, r, done = env.step(a)              # 1 歩進める
            target = r + gamma * Q[s2].max()       # 目標（終端なら Q は 0）
            Q[s, a] += alpha * (target - Q[s, a])  # 目標に少し寄せる
            s = s2
        if on_episode:
            on_episode(ep, Q)                      # 途中経過の記録用
    return Q
# [end: q-learning]


def greedy_value(Q, P):
    """Q の貪欲方策を、モデルを使って正確に評価したときの V(スタート)。"""
    policy = {s: np.eye(4)[int(np.argmax(Q[s]))] for s in STATES}
    V, _ = policy_evaluation(P, policy, GAMMA, tol=1e-6)
    return V[START]


if __name__ == "__main__":
    from figdata import write_q, write_arrows
    OUT = "ch20/out"
    P = build_model()
    Qstar = np.load(f"{OUT}/q_star.npy")
    n_episodes, every = 2000, 20

    # 学習曲線: 20 回の実行の平均。貪欲方策の真の価値を 20 エピソードごとに測る
    curves = {}
    for eps in [0.0, 0.1, 0.3]:
        vals = np.zeros((20, n_episodes // every))
        for run in range(20):
            def record(ep, Q):
                if (ep + 1) % every == 0:
                    vals[run, ep // every] = greedy_value(Q, P)
            env = GridWorld(seed=run)
            rng = np.random.default_rng(20000 + run)
            q_learning(env, n_episodes, 0.1, GAMMA, eps, rng, record)
        curves[eps] = vals
    with open(f"{OUT}/q_learning.csv", "w") as f:
        f.write("episode,eps0,eps01,eps03\n")
        for i in range(n_episodes // every):
            row = [curves[e][:, i].mean() for e in [0.0, 0.1, 0.3]]
            f.write(f"{(i + 1) * every}," +
                    ",".join(f"{x:.4f}" for x in row) + "\n")

    sys.stdout = open(f"{OUT}/q_learning_curve.txt", "w")
    print(f"最適方策の価値 V*(8) = {Qstar[START].max():.4f}")
    for eps in [0.0, 0.1, 0.3]:
        v = curves[eps]
        print(f"ε = {eps}: 貪欲方策の V(8) の平均  "
              f"100 回後 {v[:, 4].mean():+.3f}、"
              f"500 回後 {v[:, 24].mean():+.3f}、2000 回後 "
              f"{v[:, -1].mean():+.3f}（最小 {v[:, -1].min():+.3f}）")

    sys.stdout = open(f"{OUT}/q_learning.txt", "w")
    # [snippet: q-learning-demo]
    env, rng = GridWorld(seed=0), np.random.default_rng(20000)
    Q = q_learning(env, 2000, alpha=0.1, gamma=GAMMA, eps=0.1, rng=rng)
    print("状態    上      右      下      左    貪欲な行動")
    for s in STATES:
        print(f"{s:>3}  " + "  ".join(f"{q:+.3f}" for q in Q[s]) +
              f"    {ACTIONS[int(np.argmax(Q[s]))]}")
    # [end: q-learning-demo]
    write_q(f"{OUT}/fig_q_learned.tex", Q)
    write_arrows(f"{OUT}/fig_pi_learned.tex",
                 {s: int(np.argmax(Q[s])) for s in STATES})

    sys.stdout = open(f"{OUT}/q_compare.txt", "w")
    err = np.abs(Q - Qstar)[STATES]
    print(f"|Q - Q*| の最大 {err.max():.3f}、平均 {err.mean():.3f}")
    agree = sum(int(np.argmax(Q[s])) == int(np.argmax(Qstar[s]))
                for s in STATES)
    print(f"貪欲な行動が最適方策と一致する状態: {agree} / {len(STATES)}")
    print(f"学習した貪欲方策の V(8) = {greedy_value(Q, P):+.4f}"
          f"（最適 {Qstar[START].max():+.4f}）")
