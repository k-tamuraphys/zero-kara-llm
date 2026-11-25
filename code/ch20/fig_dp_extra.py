"""動的計画法の節と MDP の節に足した図のデータ（本文には載せない）。

1. 方策を決めたときの「マスからマスへ」の確率 Σ_a π(a|s) P(s'|s,a)
   （ランダム方策と最適方策）。矢印の太さで描く。
2. 価値反復の 1〜5 回目の表。前の回から新しく 0 でなくなったマスに枠。
3. 方策反復で、評価に使った書き換えの回数の合計。
4. 同じ 1 本の軌跡を 3 回くり返し経験したときの、MC と TD の表。

実行: cd code && python ch20/fig_dp_extra.py > ch20/out/fig_dp_extra.txt
"""
import sys
import numpy as np
sys.path.insert(0, "ch20")
from gridworld import N_STATES, STATES, COLS, build_model, uniform_policy
from policy_eval import GAMMA, policy_evaluation
from value_iteration import value_iteration, greedy_policy
from figdata import write_values
from mc_td import mc_update, td_update

OUT = "ch20/out"


def chain(P, policy):
    """方策 policy のもとでの、マスからマスへの確率の表 M[s][s']。"""
    M = np.zeros((N_STATES, N_STATES))
    for s in STATES:
        for a in range(4):
            for p, s2, _ in P[s][a]:
                M[s, s2] += policy[s][a] * p
    return M


def write_chain(path, M):
    """隣のマスへ移る確率を、太さ付きの矢印として書く（とどまる分は除く）。"""
    lines = []
    for s in STATES:
        for s2 in range(N_STATES):
            if s2 != s and M[s, s2] > 0:
                w = 0.3 + 3.2 * M[s, s2]           # 太さ（pt）
                lines.append(f"\\draw[->,bookblue,line width={w:.2f}pt,"
                             f"bend left=14] (c{s}) to (c{s2});")
    open(path, "w").write("\n".join(lines) + "\n")


def show_chain(name, M):
    print(f"{name}: 各マスから移る先と確率（0 のものは省く）")
    for s in STATES:
        row = "、".join(f"{s2}: {M[s, s2]:.3f}"
                       for s2 in range(N_STATES) if M[s, s2] > 0)
        print(f"  {s:>2} → {row}")


if __name__ == "__main__":
    P = build_model()

    # 1. 方策を決めると、マスの列の確率が決まる
    V_star, history = value_iteration(P, GAMMA)
    pi_star = greedy_policy(P, V_star, GAMMA)
    for name, pol in [("random", uniform_policy()), ("star", pi_star)]:
        M = chain(P, pol)
        write_chain(f"{OUT}/fig_chain_{name}.tex", M)
        show_chain("ランダム方策" if name == "random" else "最適方策", M)

    # 2. 価値反復のコマ送り（新しく値が入ったマスに枠）
    prev = np.zeros(N_STATES)
    for k in range(1, 6):
        V = history[k - 1][0]
        path = f"{OUT}/fig_vi_frame{k}.tex"
        write_values(path, V)
        new = [s for s in STATES if abs(V[s]) > 5e-3 and abs(prev[s]) <= 5e-3]
        with open(path, "a") as f:
            for s in new:
                row, col = divmod(s, COLS)
                f.write(f"\\rlbnew{{{col}}}{{{row}}}\n")
        print(f"価値反復 {k} 回目: 新しく値が入ったマス {new}")
        prev = V

    # 3. 方策反復の手間（評価の書き換えの回数）
    policy, total = uniform_policy(), 0
    for k in range(10):
        V, sweeps = policy_evaluation(P, policy, GAMMA)
        total += sweeps
        new_policy = greedy_policy(P, V, GAMMA)
        print(f"方策反復: π_{k} の評価に {sweeps} 回")
        if all((new_policy[s] == policy[s]).all() for s in STATES):
            break
        policy = new_policy
    print(f"方策反復の評価の書き換えの合計: {total} 回"
          f"（価値反復は {len(history)} 回）")

    # 4. 同じ軌跡（8→9→10→6→2→ゴール）を 3 回経験したときの表
    traj = [(8, 1, 0.0), (9, 1, 0.0), (10, 1, 0.0), (6, 0, 0.0), (2, 1, 1.0)]
    V_mc, V_td = np.zeros(N_STATES), np.zeros(N_STATES)
    print("くり返し  方法  " + "  ".join(f"V({s})" for s, _, _ in traj))
    cells = []                           # \rlbft{行}{列}{値}{色}{変化}
    for k in range(1, 4):
        for i, (name, V, update) in enumerate(
                [("MC", V_mc, mc_update), ("TD", V_td, td_update)]):
            old = V.copy()
            update(V, traj, 0.05, GAMMA)
            print(f"  {k} 回目   {name}   "
                  + "  ".join(f"{V[s]:.4f}" for s, _, _ in traj))
            for j, (s, _, _) in enumerate(traj):
                moved = int(abs(V[s] - old[s]) > 5e-5)
                k_col = min(int(round(V[s] * 400)), 70)
                cells.append(f"\\rlbft{{{k - 1}}}{{{6 * i + j}}}"
                             f"{{{V[s]:.4f}}}{{bookgreen!{k_col}}}"
                             f"{{{moved}}}")
    open(f"{OUT}/fig_mc_td_frames.tex", "w").write("\n".join(cells) + "\n")
