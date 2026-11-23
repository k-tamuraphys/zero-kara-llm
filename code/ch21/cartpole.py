"""倒立振子（CartPole）の環境を自前で実装する。

物理モデルと定数は gymnasium の CartPole-v1 と同じ（Barto ら 1983 に由来）。
n 個の環境をまとめて（ベクトル化して）同時に動かせる。
"""
import math
import torch


# [snippet: cartpole]
class CartPole:
    g, m_cart, m_pole, l = 9.8, 1.0, 0.1, 0.5   # 重力・質量 2 つ・棒の半長
    force, dt = 10.0, 0.02                      # 押す力 [N]・時間刻み [s]
    x_max, th_max = 2.4, 12 * math.pi / 180     # 失敗の判定（位置・角度）
    max_steps = 500                             # 500 ステップ立てば成功

    def __init__(self, n=1):
        self.n = n

    def reset(self):
        # 状態 s = (台車の位置 x, 速度 x', 棒の角度 θ, 角速度 θ')
        self.s = (torch.rand(self.n, 4) - 0.5) * 0.1   # 一様乱数 ±0.05
        return self.s.clone()

    # [snippet: cartpole-step]
    def step(self, a):                  # a: (n,) の 0（左へ押す）か 1（右）
        x, xd, th, thd = self.s.unbind(1)
        F = self.force * (2 * a.float() - 1)
        M = self.m_cart + self.m_pole
        cos, sin = torch.cos(th), torch.sin(th)
        tmp = (F + self.m_pole * self.l * thd ** 2 * sin) / M
        thacc = (self.g * sin - cos * tmp) / (
            self.l * (4 / 3 - self.m_pole * cos ** 2 / M))
        xacc = tmp - self.m_pole * self.l * thacc * cos / M
        # オイラー法で dt だけ時間を進める
        self.s = torch.stack([x + self.dt * xd, xd + self.dt * xacc,
                              th + self.dt * thd, thd + self.dt * thacc], 1)
        done = (self.s[:, 0].abs() > self.x_max) | \
               (self.s[:, 2].abs() > self.th_max)
        reward = torch.ones(self.n)     # 立っていれば 1 ステップごとに +1
        return self.s.clone(), reward, done
    # [end: cartpole-step]
# [end: cartpole]


if __name__ == "__main__":
    # gymnasium の CartPole-v1 と、同じ初期状態・同じ行動列で比べる
    import gymnasium as gym
    torch.manual_seed(0)
    genv = gym.make("CartPole-v1")
    env = CartPole(1)
    max_diff, steps, agree = 0.0, 0, 0
    for ep in range(10):
        s_g, _ = genv.reset(seed=ep)
        env.s = torch.tensor(s_g)[None]           # 同じ初期状態から始める
        while True:
            a = int(torch.randint(0, 2, ()))
            s_g, _, term, trunc, _ = genv.step(a)
            s, _, done = env.step(torch.tensor([a]))
            diff = (s[0] - torch.tensor(s_g)).abs().max().item()
            max_diff, steps = max(max_diff, diff), steps + 1
            if term or trunc or done[0]:
                agree += int(term == bool(done[0]))
                break
    print(f"ランダムな行動で 10 エピソード（計 {steps} ステップ）を比較")
    print(f"終了の判定が一致した回数: {agree}/10")
    print(f"状態の差の最大値: {max_diff:.1e}")
    # ランダム方策の平均エピソード長
    env = CartPole(1000)
    env.reset()
    alive = torch.ones(1000, dtype=torch.bool)
    length = torch.zeros(1000)
    for _ in range(500):
        _, _, done = env.step(torch.randint(0, 2, (1000,)))
        length += alive.float()
        alive &= ~done
    print(f"ランダム方策の平均エピソード長（1000 回）: {length.mean():.1f}")
