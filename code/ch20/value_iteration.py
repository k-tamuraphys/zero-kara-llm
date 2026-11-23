"""価値反復と方策反復: 遷移確率が分かっているときに最適方策を求める。

実行: cd code && python ch20/value_iteration.py
"""
import sys
import numpy as np
sys.path.insert(0, "ch20")
from gridworld import N_STATES, STATES, ACTIONS, START, build_model
from gridworld import uniform_policy
from policy_eval import GAMMA, q_from_v, policy_evaluation, show


# [snippet: value-iteration]
# [snippet: vi]
def value_iteration(P, gamma, tol=1e-8):
    """ベルマン最適方程式を繰り返し当てはめて V* を求める。"""
    V = np.zeros(N_STATES)
    history = []                         # 途中経過（図を描くため）
    for sweep in range(1, 10000):
        V_new = np.zeros(N_STATES)
        for s in STATES:                 # 方策評価との違いは max だけ
            V_new[s] = max(q_from_v(P, V, s, a, gamma) for a in range(4))
        diff = np.abs(V_new - V).max()
        V = V_new
        history.append((V.copy(), diff))
        if diff < tol:
            break
    return V, history
# [end: vi]


def greedy_policy(P, V, gamma):
    """価値 V に対する貪欲方策。各状態で Q が最大の行動に確率 1。"""
    policy = {}
    for s in STATES:
        q = [q_from_v(P, V, s, a, gamma) for a in range(4)]
        policy[s] = np.eye(4)[int(np.argmax(q))]   # one-hot の確率
    return policy
# [end: value-iteration]


# [snippet: policy-iteration]
def policy_iteration(P, gamma):
    """方策評価と方策改善を、方策が変わらなくなるまで交互に行う。"""
    policy = uniform_policy()            # ランダム方策から始める
    for k in range(100):
        V, _ = policy_evaluation(P, policy, gamma)       # 評価
        new_policy = greedy_policy(P, V, gamma)          # 改善
        changed = sum((new_policy[s] != policy[s]).any() for s in STATES)
        print(f"π_{k}: V(8) = {V[START]:+.4f}, V(10) = {V[10]:+.4f}, "
              f"改善で行動が変わる状態 {changed} 個")
        if changed == 0:                 # 変わらなければ最適
            break
        policy = new_policy
    return policy, V
# [end: policy-iteration]


def arrows(policy):
    return {s: int(np.argmax(policy[s])) for s in STATES}


if __name__ == "__main__":
    from figdata import write_values, write_arrows, write_q
    OUT = "ch20/out"
    P = build_model()

    sys.stdout = open(f"{OUT}/value_iteration.txt", "w")
    # [snippet: value-iteration-demo]
    V, history = value_iteration(P, GAMMA)
    print(f"{len(history)} 回の書き換えで収束")
    show(V)
    pi = greedy_policy(P, V, GAMMA)
    print("最適方策:", {s: ACTIONS[int(np.argmax(pi[s]))] for s in pi})
    # [end: value-iteration-demo]

    # 図: 途中経過、V*、最適方策、収束の様子
    for k in [1, 2, 3, 5]:
        write_values(f"{OUT}/fig_vi_sweep{k}.tex", history[k - 1][0])
    write_values(f"{OUT}/fig_v_star.tex", V)
    write_arrows(f"{OUT}/fig_pi_star.tex", arrows(pi))
    Qstar = np.zeros((N_STATES, 4))
    for s in STATES:
        Qstar[s] = [q_from_v(P, V, s, a, GAMMA) for a in range(4)]
    write_q(f"{OUT}/fig_q_star.tex", Qstar)
    np.save(f"{OUT}/q_star.npy", Qstar)
    _, sw = policy_evaluation(P, uniform_policy(), GAMMA)
    with open(f"{OUT}/vi_convergence.csv", "w") as f:
        f.write("sweep,diff\n")
        for k, (_, d) in enumerate(history, 1):
            f.write(f"{k},{d:.3e}\n")

    sys.stdout = open(f"{OUT}/q_star.txt", "w")
    print("状態   上      右      下      左")
    for s in STATES:
        print(f"{s:>3}  " + "  ".join(f"{q:+.3f}" for q in Qstar[s]))

    sys.stdout = open(f"{OUT}/policy_iteration.txt", "w")
    # [snippet: policy-iteration-demo]
    pi2, V2 = policy_iteration(P, GAMMA)
    # [end: policy-iteration-demo]
    sys.stdout = open(f"{OUT}/policy_iteration_check.txt", "w")
    same = all((pi2[s] == pi[s]).all() for s in STATES)
    print("価値反復と同じ方策か:", same)
    print(f"価値の差の最大値: {np.abs(V2 - V).max():.1e}")
    # 1 回改善しただけの方策（図用）
    V0, _ = policy_evaluation(P, uniform_policy(), GAMMA)
    pi1 = greedy_policy(P, V0, GAMMA)
    write_arrows(f"{OUT}/fig_pi_1.tex", arrows(pi1))
    V1, _ = policy_evaluation(P, pi1, GAMMA)
    write_values(f"{OUT}/fig_v_pi1.tex", V1)
