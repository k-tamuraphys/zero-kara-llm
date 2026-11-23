r"""図のためのデータを TikZ の命令として書き出す補助関数（本文には載せない）。

章の TeX 側で \rlbcell などの描画マクロを定義し、ここで作った .tex を
\input する。図の数値を、スクリプトの出力と必ず一致させるための仕組み。
"""
from gridworld import N_STATES, COLS, GOAL, PIT, WALL

ANGLES = [90, 0, 270, 180]               # 行動 上・右・下・左 の矢印の角度


def color(v, scale=70):
    """値 v を色に。正は緑、負は赤。絶対値が大きいほど濃い。"""
    k = min(int(round(abs(v) * scale)), scale)
    return ("bookgreen" if v >= 0 else "bookaccent") + f"!{k}"


def _fixed(s):
    row, col = divmod(s, COLS)
    name = {WALL: "rlbwall", GOAL: "rlbgoal", PIT: "rlbpit"}[s]
    return f"\\{name}{{{col}}}{{{row}}}"


def write_values(path, V, fmt="{:+.2f}"):
    """状態価値の格子。\\rlbcell{列}{行}{数値}{色}。"""
    lines = []
    for s in range(N_STATES):
        row, col = divmod(s, COLS)
        if s in (WALL, GOAL, PIT):
            lines.append(_fixed(s))
        else:
            lines.append(f"\\rlbcell{{{col}}}{{{row}}}"
                         f"{{{fmt.format(V[s])}}}{{{color(V[s])}}}")
    open(path, "w").write("\n".join(lines) + "\n")


def write_arrows(path, greedy):
    """決定的な方策の矢印。\\rlbarrow{列}{行}{角度}。"""
    lines = []
    for s in range(N_STATES):
        row, col = divmod(s, COLS)
        if s in (WALL, GOAL, PIT):
            lines.append(_fixed(s))
        else:
            lines.append(f"\\rlbarrow{{{col}}}{{{row}}}"
                         f"{{{ANGLES[greedy[s]]}}}")
    open(path, "w").write("\n".join(lines) + "\n")


def _num(v):
    """図の小さな三角形に収まるように、正の符号を省いた 2 桁の数にする。"""
    text = f"{abs(v):.2f}"
    return "$-$" + text if v < 0 and text != "0.00" else text


def write_q(path, Q):
    """行動価値の格子。\\rlbq{列}{行}{行動}{数値}{色}{最大なら 1}。"""
    lines = []
    for s in range(N_STATES):
        row, col = divmod(s, COLS)
        if s in (WALL, GOAL, PIT):
            lines.append(_fixed(s))
            continue
        best = int(max(range(4), key=lambda a: Q[s][a]))
        for a in range(4):
            lines.append(f"\\rlbq{{{col}}}{{{row}}}{{{a}}}"
                         f"{{{_num(Q[s][a])}}}{{{color(Q[s][a])}}}"
                         f"{{{int(a == best)}}}")
    open(path, "w").write("\n".join(lines) + "\n")
