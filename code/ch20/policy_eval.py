"""反復方策評価: ベルマン期待方程式を繰り返し当てはめて V^π を求める。

実行: cd code && python ch20/policy_eval.py
"""
import sys
import numpy as np
sys.path.insert(0, "ch20")
from gridworld import (N_STATES, STATES, ACTIONS, COLS, GOAL, PIT, WALL,
                       build_model, uniform_policy)

GAMMA = 0.9


# [snippet: policy-eval]
def q_from_v(P, V, s, a, gamma):
    """状態 s で行動 a を選んだときの価値: Σ P(s'|s,a) [r + γ V(s')]"""
    return sum(p * (r + gamma * V[s2]) for p, s2, r in P[s][a])


def policy_evaluation(P, policy, gamma, tol=1e-8):
    """方策 policy の状態価値 V^π を、表の書き換えの繰り返しで求める。"""
    V = np.zeros(N_STATES)               # すべて 0 から始める
    for sweep in range(1, 10000):
        V_new = np.zeros(N_STATES)       # 終端状態（と壁）は 0 のまま
        for s in STATES:
            V_new[s] = sum(policy[s][a] * q_from_v(P, V, s, a, gamma)
                           for a in range(4))
        diff = np.abs(V_new - V).max()   # 表がどれだけ変わったか
        V = V_new
        if diff < tol:                   # ほとんど変わらなくなったら終了
            break
    return V, sweep
# [end: policy-eval]


def show(V):
    """価値の表を格子の形に並べて表示する。"""
    for row in range(N_STATES // COLS):
        cells = []
        for col in range(COLS):
            s = row * COLS + col
            cells.append("  (壁) " if s == WALL else
                         "(ゴール)" if s == GOAL else
                         "  (穴) " if s == PIT else f"{V[s]:+.3f} ")
        print("  ".join(cells))


if __name__ == "__main__":
    from figdata import write_values
    OUT = "ch20/out"
    P = build_model()

    sys.stdout = open(f"{OUT}/policy_eval.txt", "w")
    # [snippet: policy-eval-demo]
    V, sweeps = policy_evaluation(P, uniform_policy(), GAMMA)
    print(f"{sweeps} 回の書き換えで収束")
    show(V)
    # [end: policy-eval-demo]
    write_values(f"{OUT}/fig_v_random.tex", V)

    sys.stdout = open(f"{OUT}/bellman_check.txt", "w")
    # [snippet: bellman-check]
    s = 2                                # ゴールの左隣
    total = 0.0
    for a in range(4):
        q = q_from_v(P, V, s, a, GAMMA)
        total += 0.25 * q
        print(f"Q(2, {ACTIONS[a]}) = {q:+.4f}")
    print(f"右辺: 0.25 × (4 つの和) = {total:+.4f}")
    print(f"左辺: V(2)             = {V[s]:+.4f}")
    # [end: bellman-check]
