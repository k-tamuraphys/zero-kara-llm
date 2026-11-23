"""モンテカルロ法と TD 学習: 遷移確率を知らずに、経験から V^π を推定する。

実行: cd code && python ch20/mc_td.py
"""
import sys
import numpy as np
sys.path.insert(0, "ch20")
from gridworld import (N_STATES, STATES, GridWorld, build_model,
                       uniform_policy, run_episode, returns)
from policy_eval import GAMMA, policy_evaluation, show


# [snippet: mc-td]
def mc_update(V, traj, alpha, gamma):
    """モンテカルロ法: エピソードが終わってから、実際の収益 G_t に寄せる。"""
    G = returns(traj, gamma)
    for t, (s, a, r) in enumerate(traj):
        V[s] += alpha * (G[t] - V[s])


def td_update(V, traj, alpha, gamma):
    """TD 学習: 1 歩ごとに、r_t + γ V(s_{t+1}) に寄せる。"""
    for t, (s, a, r) in enumerate(traj):
        last = (t == len(traj) - 1)
        v_next = 0.0 if last else V[traj[t + 1][0]]   # 終端状態の価値は 0
        delta = r + gamma * v_next - V[s]             # TD 誤差 δ_t
        V[s] += alpha * delta
# [end: mc-td]


def rmse(V, V_true):
    """真の価値との誤差（二乗平均平方根）。終端状態と壁は除く。"""
    return np.sqrt(np.mean((V[STATES] - V_true[STATES]) ** 2))


def experiment(alpha, n_runs=100, n_episodes=300):
    """同じエピソードを両方の方法に与え、誤差の推移を平均する。"""
    V_true, _ = policy_evaluation(build_model(), uniform_policy(), GAMMA)
    err = np.zeros((2, n_episodes))
    for run in range(n_runs):
        env = GridWorld(seed=run)
        rng = np.random.default_rng(10000 + run)
        V_mc, V_td = np.zeros(N_STATES), np.zeros(N_STATES)
        for ep in range(n_episodes):
            traj = run_episode(env, uniform_policy(), rng)
            mc_update(V_mc, traj, alpha, GAMMA)
            td_update(V_td, traj, alpha, GAMMA)
            err[0, ep] += rmse(V_mc, V_true) / n_runs
            err[1, ep] += rmse(V_td, V_true) / n_runs
    return err, V_mc, V_td, V_true


if __name__ == "__main__":
    from figdata import write_values
    OUT = "ch20/out"
    sys.stdout = open(f"{OUT}/mc_td.txt", "w")
    alphas, res = [0.02, 0.05, 0.1], {}
    print("  α    10 回目 MC / TD    300 回目 MC / TD")
    for alpha in alphas:
        res[alpha] = experiment(alpha)
        err = res[alpha][0]
        print(f"{alpha:<5}    {err[0, 9]:.3f} / {err[1, 9]:.3f}"
              f"       {err[0, -1]:.3f} / {err[1, -1]:.3f}")
    with open(f"{OUT}/mc_td.csv", "w") as f:
        f.write("episode,mc002,td002,mc005,td005,mc01,td01\n")
        for ep in range(res[0.05][0].shape[1]):
            row = [res[a][0][i, ep] for a in alphas for i in range(2)]
            f.write(f"{ep + 1}," + ",".join(f"{x:.4f}" for x in row) + "\n")

    # α = 0.05 の最後の 1 回の実行で得た表（図用）
    _, V_mc, V_td, V_true = res[0.05]
    write_values(f"{OUT}/fig_v_mc.tex", V_mc)
    write_values(f"{OUT}/fig_v_td.tex", V_td)
    sys.stdout = open(f"{OUT}/mc_td_values.txt", "w")
    print("真の価値（反復方策評価）")
    show(V_true)
    print("モンテカルロ法（α = 0.05、300 エピソード、最後の 1 回の実行）")
    show(V_mc)
    print("TD 学習（同じエピソード）")
    show(V_td)
