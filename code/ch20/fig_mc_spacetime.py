"""モンテカルロ法の時空図のデータ（本文には載せない。図 20.x 用）。

横軸に時刻 t、縦軸にマスをとり、ランダム方策の軌跡を何本か重ねる。
状態 9 を通った時刻ごとに「そこから先の収益 G_t」を集めて平均する。
あわせて、エピソードを増やしたときの平均の推移を csv に書く。

実行: cd code && python ch20/fig_mc_spacetime.py \
          > ch20/out/fig_mc_spacetime.txt
"""
import sys
import numpy as np
sys.path.insert(0, "ch20")
from gridworld import GridWorld, uniform_policy, run_episode, returns
from policy_eval import GAMMA, policy_evaluation
from gridworld import build_model

OUT = "ch20/out"
TARGET = 9                               # 平均を取るマス
ROWS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]   # 縦軸の並び（壁 5 を除く）
COLORS = ["bookblue", "bookorange", "bookgreen", "bookpurple"]
MAX_LEN = 12                             # 図に描くのは短いエピソードだけ


def pick_episodes(env, rng, n_pick):
    """先頭から順に動かし、短くて TARGET を通るものを n_pick 本選ぶ。"""
    picked, ep = [], 0
    while len(picked) < n_pick:
        traj = run_episode(env, uniform_policy(), rng)
        states = [s for s, _, _ in traj]
        if len(traj) <= MAX_LEN and TARGET in states:
            picked.append((ep, traj))
        ep += 1
    return picked


def write_tikz(path, picked):
    """軌跡の折れ線と、TARGET を通った点の丸を TikZ の命令で書く。"""
    lines = []
    for k, (ep, traj) in enumerate(picked):
        c, dy = COLORS[k], 0.12 * (k - 1.5)    # 線が重ならないように少しずらす
        last = 3 if traj[-1][2] > 0 else 7     # 終端状態（ゴールか穴）
        ys = [s for s, _, _ in traj] + [last]
        pts = " -- ".join(f"({t},{-ROWS.index(s) + dy:.2f})"
                          for t, s in enumerate(ys))
        lines.append(f"\\draw[{c},thick] {pts};")
        T = len(traj)
        lines.append(f"\\fill[{c}] ({T},{-ROWS.index(last) + dy:.2f})"
                     " ++(-2pt,-2pt) rectangle ++(4pt,4pt);")
        for t, s in enumerate(ys[:-1]):
            if s == TARGET:
                y = -ROWS.index(s) + dy
                lines.append(f"\\draw[{c},fill=white,thick] "
                             f"({t},{y:.2f}) circle (2.2pt);")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    V_true, _ = policy_evaluation(build_model(), uniform_policy(), GAMMA)
    env, rng = GridWorld(seed=1), np.random.default_rng(2001)
    picked = pick_episodes(env, rng, 4)
    write_tikz(f"{OUT}/fig_mc_spacetime.tex", picked)
    print(f"状態 {TARGET} を通った時刻と、そこから先の収益 G_t")
    samples = []
    for k, (ep, traj) in enumerate(picked):
        G = returns(traj, GAMMA)
        end = "ゴール" if traj[-1][2] > 0 else "穴"
        got = [(t, G[t]) for t, (s, _, _) in enumerate(traj) if s == TARGET]
        samples += [g for _, g in got]
        print(f"試行 {k + 1}（{ep + 1} 本目、{len(traj)} 歩で{end}）: "
              + "、".join(f"t={t} で {g:+.3f}" for t, g in got))
    print(f"集めた {len(samples)} 個の平均: {np.mean(samples):+.3f}"
          f"（正しい V(9) = {V_true[TARGET]:+.3f}）")
    # 図の右に置く「集めた値」の一覧（TikZ のノード）
    rows = []
    for k, (ep, traj) in enumerate(picked):
        G = returns(traj, GAMMA)
        vals = "、".join(f"${G[t]:+.2f}$"
                        for t, (s, _, _) in enumerate(traj) if s == TARGET)
        rows.append(f"\\textcolor{{{COLORS[k]}}}{{試行 {k + 1}}}: {vals}")
    rows.append(f"{len(samples)} 個の平均: ${np.mean(samples):+.2f}$")
    rows.append(f"正しい $V^\\pi(9)={V_true[TARGET]:+.3f}$")
    open(f"{OUT}/fig_mc_spacetime_list.tex", "w").write(
        "\\\\\n".join(rows) + "\n")

    # エピソードを増やしたときの見積もりの推移（新しい環境と乱数で）
    env, rng = GridWorld(seed=2), np.random.default_rng(2002)
    total, n, V_alpha = 0.0, 0, 0.0
    with open(f"{OUT}/fig_mc_spacetime.csv", "w") as f:
        f.write("episode,mean,alpha005,visits\n")
        for ep in range(1, 2001):
            traj = run_episode(env, uniform_policy(), rng)
            G = returns(traj, GAMMA)
            for t, (s, _, _) in enumerate(traj):
                if s == TARGET:          # 通るたびに 1 個のデータとして数える
                    total, n = total + G[t], n + 1
                    V_alpha += 0.05 * (G[t] - V_alpha)
            if n > 0:
                f.write(f"{ep},{total / n:.4f},{V_alpha:.4f},{n}\n")
            if ep in (10, 100, 1000, 2000):
                print(f"{ep:>5} エピソード: データ {n:>5} 個、"
                      f"平均 {total / n:+.3f}、α=0.05 の更新 {V_alpha:+.3f}")
