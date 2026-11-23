"""reward-to-go で崩れた種（1 と 2）の方策を調べる。"""
import torch
from reinforce import train, rollout

torch.set_num_threads(2)
# [snippet: collapse]
for seed in [1, 2]:
    _, policy, _ = train("rtg", seed)               # reinforce.py と同じ設定
    S, A, R, M = rollout(policy, 100)
    m = M.bool()                                   # エピソードの内側だけ
    dist = policy.dist(S)
    print(f"種 {seed}: 平均長 {M.sum(0).mean():4.1f}"
          f"  エントロピー {dist.entropy()[m].mean():.3f}（最大 0.693）"
          f"  右へ押した割合 {A[m].float().mean():.2f}")
# [end: collapse]
