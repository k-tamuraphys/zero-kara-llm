"""第 22 章が共有する道具：生成（ロールアウト）、報酬、対数確率。"""
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmbook.corpus import DATA_DIR
from llmbook.tokenizer import BPETokenizer
from llmbook.chat import render_chat, make_chat_example, chat_accuracy
from llmbook.train import load_checkpoint, save_checkpoint

CKPT_DIR = DATA_DIR.parent / "checkpoints"
OUT_DIR = Path(__file__).resolve().parent / "out"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)  # CPU を他の作業と分け合う

tok = BPETokenizer.load(DATA_DIR / "bpe4096.json")
END = tok.special_tokens["<|end|>"]
PAD = tok.special_tokens["<|pad|>"]


def load_sft():
    """第 16 章の計算アシスタント sft.pt を読み込む。"""
    return load_checkpoint(CKPT_DIR / "sft.pt", device)


def prompt_ids(question: str) -> list[int]:
    msgs = [{"role": "user", "content": question}]
    return tok.encode(render_chat(msgs, add_generation_prompt=True))


# [snippet: reward]
def reward(reply_ids: list[int], answer: str) -> float:
    """検証可能な報酬：応答が「答えです。」と完全一致すれば 1、違えば 0。"""
    reply = tok.decode(reply_ids).split("<|end|>")[0]
    return 1.0 if reply == f"{answer}です。" else 0.0
# [end: reward]


# [snippet: rollout]
@torch.no_grad()
def rollout(model, prompt: list[int], G: int, max_new: int = 8,
            temperature: float = 1.0):
    """同じプロンプトから G 本の応答をサンプリングする。
    返り値 resp (G, T') は生成したトークン、mask (G, T') は
    <|end|> までの「モデルが書いた」位置なら 1、その後ろは 0。"""
    model.eval()
    idx = torch.tensor([prompt] * G, device=device)      # (G, P)
    out = model.generate(idx, max_new, temperature=temperature,
                         stop_token=END)
    resp = out[:, len(prompt):]                          # (G, T')
    is_end = (resp == END).long()
    # <|end|> より後ろ（すでに終わった応答の続き）は 0 にする
    after_end = (is_end.cumsum(dim=1) - is_end) > 0
    mask = (~after_end).float()
    return resp, mask
# [end: rollout]


# [snippet: token-logprobs]
def token_logprobs(model, prompt: list[int], resp: torch.Tensor):
    """応答の各トークンの対数確率 log π(o_t | q, o_<t) を (G, T') で返す。"""
    G, Tr = resp.shape
    idx = torch.cat([torch.tensor([prompt] * G, device=device), resp], 1)
    logits, _ = model(idx[:, :-1])                       # (G, P+T'-1, V)
    logits = logits[:, len(prompt) - 1:]                 # 応答を予測する位置
    logp = F.log_softmax(logits, dim=-1)                 # (G, T', V)
    return logp.gather(2, resp.unsqueeze(2)).squeeze(2)  # (G, T')
# [end: token-logprobs]


def reply_text(resp_row: torch.Tensor) -> str:
    return tok.decode(resp_row.tolist()).split("<|end|>")[0]


def accuracy(model, n: int = 200) -> float:
    """第 16 章と同じ物差し（貪欲生成・200 問・種 12345）で正答率を測る。"""
    return chat_accuracy(model, tok, n_problems=n, device=device)
