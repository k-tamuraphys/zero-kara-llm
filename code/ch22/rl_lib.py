"""rl_long.py と rl_ppo.py（と第 24 章）が使い回す道具：
束ねた生成、束ねた対数確率、難しい問題、固定の評価セット。"""
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from common import END, PAD, device, prompt_ids, reward, make_chat_example
from llmbook.chat import QUESTION_TEMPLATES


def make_hard_example(rng):
    """3 桁どうしの足し算・引き算（繰り上がり・繰り下がりが多い）。"""
    op = rng.choice("+-")
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    if op == "-" and a < b:
        a, b = b, a
    q = rng.choice(QUESTION_TEMPLATES[op]).format(a=a, b=b)
    return q, str(a + b if op == "+" else a - b)


@torch.no_grad()
def generate_grouped(model, prompts, n_each, temperature, max_new=8):
    """長さの同じプロンプトをまとめて生成する。返り値は prompts の順の
    応答トークン (n_each, max_new) のリスト。"""
    model.eval()
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    outs = [None] * len(prompts)
    for L, ids in by_len.items():
        idx = torch.tensor([prompts[i] for i in ids for _ in range(n_each)],
                           device=device)
        out = model.generate(idx, max_new, temperature=temperature,
                             stop_token=END)[:, L:]
        if out.size(1) < max_new:  # 全員が早く終えたら END で埋める
            out = F.pad(out, (0, max_new - out.size(1)), value=END)
        for k, i in enumerate(ids):
            outs[i] = out[k * n_each:(k + 1) * n_each]
    return outs


def batch_logprobs(model, seqs, P_lens, T):
    """右を埋めた (N, P+T) の列から、応答部分 T トークンの log π を返す。"""
    L = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), L), PAD, device=device)
    for k, s in enumerate(seqs):
        idx[k, :len(s)] = torch.tensor(s, device=device)
    logits, _ = model(idx[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    pos = torch.tensor(P_lens, device=device).unsqueeze(1) - 1 + \
        torch.arange(T, device=device)                      # (N, T)
    tgt = idx.gather(1, pos + 1)
    lp = logp.gather(1, pos.unsqueeze(2).expand(-1, -1, logp.size(2)))
    return lp.gather(2, tgt.unsqueeze(2)).squeeze(2)        # (N, T)


def make_eval_set(n, seed):
    rng = random.Random(seed)
    exs = [make_chat_example(rng) for _ in range(n)]
    return [(prompt_ids(m[0]["content"]), a) for m, a in exs]


EVAL_TEST = make_eval_set(1000, 777)  # 報告用（rl_long.py の 1,000 問と同じ）
EVAL_VAL = make_eval_set(500, 555)    # 最良モデルを選ぶ用


def eval_acc(model, evalset):
    """貪欲生成で、応答が「答えです。」と一致する割合。"""
    outs = generate_grouped(model, [p for p, _ in evalset], 1, 0.0)
    return sum(reward(o[0].tolist(), a)
               for o, (_, a) in zip(outs, evalset)) / len(evalset)
