"""床の滑りやすさ SLIP を変えて、最適方策と V*(8) がどう変わるかを見る。

gridworld.SLIP（左右それぞれにそれる確率）を 0, 0.1, 0.2, 0.3 に変え、
そのたびに遷移確率の表を作り直して価値反復を行う。γ = 0.9 は固定。

実行: cd code && python ch20/sweep_slip.py > ch20/out/sweep_slip.txt
"""
import sys
import numpy as np
sys.path.insert(0, "ch20")
import gridworld
from gridworld import STATES, START, ACTIONS
from policy_eval import GAMMA, q_from_v
from value_iteration import value_iteration, greedy_policy, arrows
from figdata import write_values, write_arrows

OUT = "ch20/out"

if __name__ == "__main__":
    with open(f"{OUT}/sweep_slip.csv", "w") as f:
        f.write("slip,v8,q8up,q8right,goal\n")
    for k, slip in enumerate([0.0, 0.1, 0.2, 0.3]):
        gridworld.SLIP = slip            # build_model はこの値を読む
        P = gridworld.build_model()
        V, history = value_iteration(P, GAMMA)
        pi = greedy_policy(P, V, GAMMA)
        q8 = [q_from_v(P, V, START, a, GAMMA) for a in range(4)]
        write_values(f"{OUT}/fig_slip{k}_v.tex", V)
        write_arrows(f"{OUT}/fig_slip{k}_pi.tex", arrows(pi))
        # 最適方策で動いたとき、穴ではなくゴールに着く確率
        # （報酬だけ +1/0 にした表で、γ = 1 の方策評価をする）
        goal = np.zeros(gridworld.N_STATES)
        for _ in range(5000):
            new = np.zeros_like(goal)
            for s in STATES:
                a = int(np.argmax(pi[s]))
                new[s] = sum(p * ((s2 == gridworld.GOAL) + goal[s2])
                             for p, s2, _ in P[s][a])
            goal = new
        print(f"SLIP = {slip:.1f}: V*(8) = {V[START]:+.4f}、"
              f"Q*(8,上) = {q8[0]:+.4f}、Q*(8,右) = {q8[1]:+.4f}、"
              f"ゴールに着く確率 {goal[START]:.3f}、"
              f"{len(history)} 回で収束")
        print("   方策: " + " ".join(f"{s}{ACTIONS[int(np.argmax(pi[s]))]}"
                                   for s in STATES))
        with open(f"{OUT}/sweep_slip.csv", "a") as f:
            f.write(f"{slip},{V[START]:.4f},{q8[0]:.4f},{q8[1]:.4f},"
                    f"{goal[START]:.4f}\n")
