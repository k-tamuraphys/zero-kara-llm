"""学習率スケジュールの 3 つの形を、最大値を 1 にそろえて csv に書き出す。

本書の lr_at（base.pt の設定）、DeepSeek-V3 の報告にある形、
nanochat の既定値の形を、横軸「学習の進み具合（0〜1）」で並べる。
実行: python ch13/fig_schedules.py  （一瞬）
出力: ch13/out/schedules.csv（列は x, cosine, deepseek, nanochat）
"""

import math

from llmbook.train import lr_at


def deepseek(x: float) -> float:
    """DeepSeek-V3: 14.8 兆トークンのうち 10 兆までは一定、4.3 兆で
    コサインで 10% まで下げ、3,330 億は 10%、最後の 1,670 億は 3.3%。
    （最初の 2,000 ステップのウォームアップは、この縮尺では見えない）"""
    t = x * 14.8  # 兆トークン
    if t < 10.0:
        return 1.0
    if t < 14.3:
        p = (t - 10.0) / 4.3
        return 0.1 + 0.9 * (1 + math.cos(math.pi * p)) / 2
    if t < 14.633:
        return 0.1
    return 7.3e-6 / 2.2e-4


def nanochat(x: float) -> float:
    """nanochat の既定値: 一定のあと、最後の 65% で 5% まで直線で下げる。
    （ウォームアップは 40 ステップで、この縮尺では見えない）"""
    if x < 0.35:
        return 1.0
    return 1.0 - 0.95 * (x - 0.35) / 0.65


# 段差をまっすぐ描くため、折れ目の直前の点も入れる
knots = {10 / 14.8, 14.3 / 14.8, 14.633 / 14.8 - 1e-5, 14.633 / 14.8, 0.35}
xs = sorted({i / 400 for i in range(401)} | knots)
with open("ch13/out/schedules.csv", "w") as f:
    f.write("x,cosine,deepseek,nanochat\n")
    for x in xs:
        step = min(3999, int(x * 4000))
        cos = lr_at(step, 1.0, warmup_steps=200, max_steps=4000)
        f.write(f"{x:.5f},{cos:.5f},{deepseek(x):.5f},{nanochat(x):.5f}\n")
