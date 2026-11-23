"""自作環境で学習した PPO の方策を、gymnasium の CartPole-v1 で評価する。"""
import gymnasium as gym
import torch
from reinforce import PolicyNet

# [snippet: eval-gym]
policy = PolicyNet()
policy.load_state_dict(torch.load("checkpoints/ch21_ppo.pt"))
env = gym.make("CartPole-v1")
for mode in ["sample", "greedy"]:
    lengths = []
    for ep in range(20):
        s, _ = env.reset(seed=1000 + ep)
        total, done = 0, False
        while not done:
            with torch.no_grad():
                d = policy.dist(torch.tensor(s))
            a = d.sample() if mode == "sample" else d.probs.argmax()
            s, r, terminated, truncated, _ = env.step(int(a))
            total += r
            done = terminated or truncated       # 倒れた／500 に達した
        lengths.append(int(total))
    print(f"{mode:>6}: 平均 {sum(lengths)/20:.1f}  最小 {min(lengths)}"
          f"  （20 エピソード）")
# [end: eval-gym]
