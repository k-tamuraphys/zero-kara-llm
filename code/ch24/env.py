"""第 24 章：電卓つきの計算環境と、束ねた多ターンのロールアウト。

  env.make_task(rng, kind) -> Task(question, answer, kind)
  rollout(model, tasks, temperature, eps) -> list[Traj]
第 23 章の電卓 calculator と書式（<|tool|>式<|end|><|result|>値<|end|>）を使う。
"""
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "ch23"))
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.chat import make_chat_example
from llmbook.train import load_checkpoint, save_checkpoint  # noqa: F401
from tools import calculator, make_tool_example  # 第 23 章の電卓と問題

CKPT_DIR = DATA_DIR.parent / "checkpoints"
OUT_DIR = HERE / "out"
device = "cpu"
tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
END = tok.special_tokens["<|end|>"]
PAD = tok.special_tokens["<|pad|>"]
TOOL = tok.special_tokens["<|tool|>"]
ASSIST = tok.special_tokens["<|assistant|>"]


@dataclass
class Task:
    question: str
    answer: str
    kind: str  # "add"（加減算）/ "mul"（掛け算）/ "two"（掛けて足す）


def make_task(rng: random.Random, kind: str) -> Task:
    if kind == "add":  # 第 16・22 章と同じ質問文（3 種の言い回し）
        m, a = make_chat_example(rng, ops="+-")
    else:              # 第 23 章の掛け算・掛けて足す
        m, a = make_tool_example(rng, kind)
    return Task(m[0]["content"], a, kind)


# [snippet: traj]
@dataclass
class Traj:
    task: Task
    ids: list            # 文脈の全トークン（質問 → 発話 → 結果 → ...）
    mask: list           # ids と同じ長さ。モデルが書いたトークンなら 1
    n_calls: int = 0     # 電卓を呼んだ回数
    reply: str | None = None  # 最終応答（なければ None ＝ 失敗）
    texts: list = field(default_factory=list)  # 発話の文字列（表示用）
# [end: traj]


def _sample_first(logits, temperature, eps, gen):
    """最初の役割トークンを選ぶ。確率 eps で <|tool|> と <|assistant|>
    を半々に強制する（探索の工夫）。"""
    if temperature == 0.0:
        nxt = logits.argmax(-1)
    else:
        nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1,
                                generator=gen).squeeze(1)
    if eps > 0:
        u = torch.rand(len(nxt), generator=gen)
        coin = torch.rand(len(nxt), generator=gen) < 0.5
        forced = torch.where(coin, torch.tensor(TOOL), torch.tensor(ASSIST))
        nxt = torch.where(u < eps, forced, nxt)
    return nxt


def _forward_cached(model, ids, pos, keymask, cache):
    """ids (N, T) を位置 pos (N, T) で通し、層ごとの K, V を cache に足す。
    keymask (N, S) は「見てよい位置（詰め物でない）」。KV キャッシュ（第 11 章）。"""
    x = model.tok_emb(ids) + model.pos_emb(pos)
    N, T, C = x.shape
    S = keymask.size(1)
    causal = torch.ones(T, S, dtype=torch.bool).tril(S - T)
    diag = torch.ones(T, S, dtype=torch.bool).tril(S - T) & \
        ~torch.ones(T, S, dtype=torch.bool).tril(S - T - 1)  # 自分自身
    allow = ((causal[None] & keymask[:, None, :]) | diag[None])[:, None]
    for li, blk in enumerate(model.blocks):
        a = blk.attn
        H = a.n_head
        q, k, v = a.qkv(blk.ln1(x)).split(C, dim=2)
        q, k, v = (z.view(N, T, H, C // H).transpose(1, 2) for z in (q, k, v))
        if li in cache:
            k = torch.cat([cache[li][0], k], 2)
            v = torch.cat([cache[li][1], v], 2)
        cache[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // H)
        att = att.masked_fill(~allow, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(N, T, C)
        x = x + a.proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return model.lm_head(model.ln_f(x[:, -1]))              # 末尾の予測だけ


@torch.no_grad()
def _generate_turn(model, ctxs, temperature, eps, max_new, gen):
    """文脈のリストそれぞれに「次の 1 発話」（<|end|> まで）を生成する。
    長さの違う文脈は、前に詰め物を入れて末尾をそろえる（位置の番号は詰め物の
    ぶんだけずらし、詰め物は注意から隠す）。KV キャッシュで速く回す。"""
    model.eval()
    N = len(ctxs)
    L = max(len(c) for c in ctxs)
    pad = torch.tensor([L - len(c) for c in ctxs])
    idx = torch.full((N, L), PAD)
    for k, c in enumerate(ctxs):
        idx[k, L - len(c):] = torch.tensor(c)
    pos = (torch.arange(L)[None] - pad[:, None]).clamp(min=0)
    keymask = torch.arange(L)[None] >= pad[:, None]
    cache = {}
    lg = _forward_cached(model, idx, pos, keymask, cache)
    done = torch.zeros(N, dtype=torch.bool)
    out = []
    for s in range(max_new):
        if s == 0:
            nxt = _sample_first(lg, temperature, eps, gen)
        elif temperature == 0.0:
            nxt = lg.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(lg / temperature, -1), 1,
                                    generator=gen).squeeze(1)
        nxt = torch.where(done, torch.tensor(PAD), nxt)
        out.append(nxt)
        done = done | (nxt == END)
        if done.all():
            break
        pos = pos[:, -1:] + 1
        keymask = torch.cat([keymask, torch.ones(N, 1, dtype=torch.bool)], 1)
        lg = _forward_cached(model, nxt[:, None], pos, keymask, cache)
    out = torch.stack(out, 1).tolist()
    res = []
    for t in out:
        res.append(t[:t.index(END) + 1] if END in t else t)
    return res


# [snippet: rollout]
def rollout(model, tasks, temperature=1.0, eps=0.0, max_turns=3,
            max_new=16, gen=None):
    """全タスクのエージェントループを、ターンごとに束ねて同時に進める。"""
    trajs = []
    for t in tasks:
        q = tok.encode(f"<|user|>{t.question}<|end|>")
        trajs.append(Traj(t, q, [0] * len(q)))
    active = list(range(len(trajs)))
    for turn in range(max_turns):
        if not active:
            break
        outs = _generate_turn(model, [trajs[i].ids for i in active],
                              temperature, eps if turn == 0 else 0.0,
                              max_new, gen)
        still = []
        for i, out in zip(active, outs):
            tr = trajs[i]
            tr.ids += out
            tr.mask += [1] * len(out)          # モデルが書いた部分
            text = tok.decode(out)
            tr.texts.append(text)
            if out[0] == TOOL and out[-1] == END:
                result = calculator(text.removeprefix("<|tool|>")
                                    .removesuffix("<|end|>"))
                res = tok.encode(f"<|result|>{result}<|end|>")
                tr.ids += res
                tr.mask += [0] * len(res)      # 環境が書いた部分
                tr.texts.append(f"<|result|>{result}<|end|>")
                tr.n_calls += 1
                if len(tr.ids) + max_new < 128:  # 文脈長 128 の手前で打ち切る
                    still.append(i)
            elif out[0] == ASSIST and out[-1] == END:
                tr.reply = text.removeprefix("<|assistant|>") \
                               .removesuffix("<|end|>")
            # どちらでもない発話は失敗（reply は None のまま）
        active = still
    for tr in trajs:                  # 文脈長（128）を超えた分は切り捨てる
        tr.ids, tr.mask = tr.ids[:129], tr.mask[:129]
    return trajs
# [end: rollout]


# [snippet: reward]
def reward(tr: Traj, c: float) -> float:
    """軌跡の報酬：最終応答が正しければ 1、電卓 1 回につき −c。"""
    correct = tr.reply == f"{tr.task.answer}です。"
    return float(correct) - c * tr.n_calls
# [end: reward]


def traj_logprobs(model, trajs):
    """各軌跡の全トークンの log π と mask を (N, L-1) で返す。
    短い軌跡は、後ろに詰め物を入れて長さをそろえる。"""
    L = max(len(t.ids) for t in trajs)
    idx = torch.full((len(trajs), L), PAD)
    m = torch.zeros(len(trajs), L)
    for k, t in enumerate(trajs):
        idx[k, :len(t.ids)] = torch.tensor(t.ids)
        m[k, :len(t.mask)] = torch.tensor(t.mask, dtype=torch.float)
    logits, _ = model(idx[:, :-1])
    logp = F.log_softmax(logits, -1).gather(2, idx[:, 1:, None]).squeeze(2)
    return logp, m[:, 1:]  # 位置 t の logp は「トークン t+1」の確率


def eval_sets():
    """学習に使わない固定の評価問題。加減算は第 22 章と同じ種 777 の 1,000 問。"""
    rng = random.Random(777)
    add = [make_task(rng, "add") for _ in range(1000)]
    rng = random.Random(12345)
    mul = [make_task(rng, "mul") for _ in range(300)]
    two = [make_task(rng, "two") for _ in range(300)]
    return {"add": add, "mul": mul, "two": two}


def evaluate(model, sets, bs=250):
    """貪欲（温度 0）で解かせ、種類ごとの正答率・電卓使用率・平均呼び出し回数。"""
    res, all_trajs = {}, {}
    for kind, tasks in sets.items():
        trs = []
        for s in range(0, len(tasks), bs):
            trs += rollout(model, tasks[s:s + bs], temperature=0.0)
        n = len(trs)
        res[kind] = {"acc": sum(reward(t, 0) for t in trs) / n,
                     "tool": sum(t.n_calls > 0 for t in trs) / n,
                     "calls": sum(t.n_calls for t in trs) / n}
        all_trajs[kind] = trs
    return res, all_trajs
