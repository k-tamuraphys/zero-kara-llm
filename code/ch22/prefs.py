"""選好データ：同じ質問への 2 つの応答のうち、人がどちらを好んだか。
この章では「答えは47です。」（好まれた y_w）と
「47です。」（好まれなかった y_l）の対を合成して、人の好みの代わりにする。"""
import random

import torch

from common import tok, prompt_ids, device, make_chat_example, PAD


# [snippet: make-prefs]
def make_prefs(n: int, seed: int):
    """(プロンプト, 好まれた応答, 好まれなかった応答) を n 組作る。"""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        msgs, ans = make_chat_example(rng)
        p = prompt_ids(msgs[0]["content"])
        y_w = tok.encode(f"答えは{ans}です。<|end|>")   # 丁寧に言い切る
        y_l = tok.encode(f"{ans}です。<|end|>")         # 数だけ答える
        data.append((p, y_w, y_l))
    return data
# [end: make-prefs]


def pad_batch(seqs: list[list[int]]):
    """右側をパディングで埋め、(B, T) のトークンと、各系列の長さを返す。"""
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(device), [len(s) for s in seqs]


# [snippet: seq-logprob]
def seq_logprob(model, prompts, resps):
    """各組の log π(y | x) = Σ_t log π(y_t | x, y_<t) を (B,) で返す。"""
    x, lens = pad_batch([p + y for p, y in zip(prompts, resps)])
    logits, _ = model(x[:, :-1])
    logp = torch.log_softmax(logits, -1).gather(
        2, x[:, 1:].unsqueeze(2)).squeeze(2)              # (B, T-1)
    mask = torch.zeros_like(logp)
    for i, (p, y) in enumerate(zip(prompts, resps)):
        mask[i, len(p) - 1:len(p) - 1 + len(y)] = 1.0     # 応答の部分だけ
    return (logp * mask).sum(dim=1)
# [end: seq-logprob]
