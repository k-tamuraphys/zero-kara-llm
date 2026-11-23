"""格子世界（3 行 × 4 列）: 第 20 章を通して使う小さなマルコフ決定過程。

    0   1   2   3(ゴール +1)
    4  (壁)  6   7(穴 -1)
    8   9  10  11          スタートは 8

実行: cd code && python ch20/gridworld.py
"""
import numpy as np

# [snippet: grid-def]
ROWS, COLS = 3, 4
N_STATES = ROWS * COLS                   # 状態番号 s = 4×行 + 列
GOAL, PIT, WALL, START = 3, 7, 5, 8
TERMINALS = [GOAL, PIT]                  # 終端状態
STATES = [s for s in range(N_STATES) if s not in (GOAL, PIT, WALL)]
ACTIONS = ["上", "右", "下", "左"]        # 行動 a = 0, 1, 2, 3
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # 各行動の（行, 列）の変化
SLIP = 0.1                               # 左右それぞれにそれる確率
# [end: grid-def]


# [snippet: grid-model]
def move(s, a):
    """状態 s から向き a に 1 マス進んだ先。壁か外なら動かない。"""
    row, col = divmod(s, COLS)
    row2, col2 = row + MOVES[a][0], col + MOVES[a][1]
    s2 = row2 * COLS + col2
    if not (0 <= row2 < ROWS and 0 <= col2 < COLS) or s2 == WALL:
        return s
    return s2


def reward(s2):
    """報酬は、入った先のマスで決まる。"""
    return 1.0 if s2 == GOAL else -1.0 if s2 == PIT else 0.0


# [snippet: grid-build]
def build_model():
    """P[s][a] = [(確率, 次の状態, 報酬), ...] という表を作る。"""
    P = {}
    for s in STATES:
        P[s] = []
        for a in range(4):
            probs = {}                   # 次の状態 -> 確率
            for p, d in [(1 - 2 * SLIP, a),          # 意図した向き
                         (SLIP, (a - 1) % 4),        # 左にそれる
                         (SLIP, (a + 1) % 4)]:       # 右にそれる
                s2 = move(s, d)
                probs[s2] = probs.get(s2, 0.0) + p
            P[s].append([(p, s2, reward(s2)) for s2, p in probs.items()])
    return P
# [end: grid-build]
# [end: grid-model]


# [snippet: grid-env]
class GridWorld:
    """エージェントから見た環境。中の表 P は、エージェントには見せない。"""

    def __init__(self, seed=0):
        self.P = build_model()
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.s = START
        return self.s

    def step(self, a):
        probs, nexts, rewards = zip(*self.P[self.s][a])
        i = self.rng.choice(len(probs), p=probs)   # 遷移確率に従って選ぶ
        self.s = nexts[i]
        done = self.s in TERMINALS                 # 終端状態なら終了
        return self.s, rewards[i], done
# [end: grid-env]


# [snippet: grid-rollout]
def run_episode(env, policy, rng, max_steps=1000):
    """方策 policy[s]（4 つの行動の確率）で 1 エピソード動かす。"""
    s, done, traj = env.reset(), False, []
    while not done and len(traj) < max_steps:
        a = rng.choice(4, p=policy[s])             # a_t ~ π(・|s_t)
        s2, r, done = env.step(a)                  # 環境が s_{t+1}, r_t を返す
        traj.append((s, a, r))
        s = s2
    return traj


def returns(traj, gamma):
    """各時刻の収益 G_t を、後ろから G_t = r_t + γ G_{t+1} で求める。"""
    G, out = 0.0, []
    for _, _, r in reversed(traj):
        G = r + gamma * G
        out.append(G)
    return out[::-1]
# [end: grid-rollout]


def uniform_policy():
    """どの状態でも 4 つの行動を等しい確率で選ぶ方策。"""
    return {s: np.full(4, 0.25) for s in STATES}


if __name__ == "__main__":
    import sys
    OUT = "ch20/out"
    P = build_model()

    sys.stdout = open(f"{OUT}/grid_model.txt", "w")
    # [snippet: grid-model-demo]
    for s, a in [(8, 1), (6, 0)]:
        print(f"P[{s}][{a}]（状態 {s} で「{ACTIONS[a]}」）:")
        for p, s2, r in P[s][a]:
            print(f"   確率 {p:.1f} で 状態 {s2:>2} へ、報酬 {r:+.0f}")
    # [end: grid-model-demo]

    sys.stdout = open(f"{OUT}/grid_episode.txt", "w")
    # [snippet: grid-episode-demo]
    # 環境とエージェントの乱数は、別の種にして独立にする
    env, rng = GridWorld(seed=57), np.random.default_rng(1057)
    traj = run_episode(env, uniform_policy(), rng)
    G = returns(traj, gamma=0.9)
    print(" t  状態  行動  報酬   収益 G_t")
    for t, (s, a, r) in enumerate(traj):
        print(f"{t:>2}  {s:>3}    {ACTIONS[a]}   {r:+.0f}   {G[t]:+.4f}")
    # [end: grid-episode-demo]

    # ランダム方策の収益の分布（本文の数値用）
    sys.stdout = open(f"{OUT}/grid_random_stats.txt", "w")
    env, rng = GridWorld(seed=0), np.random.default_rng(1000)
    n, G0, lens, goals = 10000, [], [], 0
    for _ in range(n):
        traj = run_episode(env, uniform_policy(), rng)
        G0.append(returns(traj, 0.9)[0])
        lens.append(len(traj))
        goals += traj[-1][2] > 0
    print(f"ランダム方策 {n} エピソード: ゴール到達 {goals / n:.3f}、"
          f"平均の長さ {np.mean(lens):.1f}、G_0 の平均 {np.mean(G0):+.4f}")
