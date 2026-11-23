"""ミニGPT をエージェントループにつなぐ。

  from mini_agent import MiniGPTLLM, agent_accuracy
  llm = MiniGPTLLM(load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt"))
  run_agent(llm, "37×48は？")  # -> ("1776です。", 文脈, 1)

実行:  python ch23/mini_agent.py  > ch23/out/mini_agent.txt   （約 2 分）
"""

import random

import torch

from agent_loop import run_agent
from common import CKPT_DIR, device, tok
from tools import calculator, make_tool_example
from llmbook.train import load_checkpoint

END_ID = tok.special_tokens["<|end|>"]


# [snippet: minigpt-llm]
class MiniGPTLLM:
    """ミニGPT を「文脈 → 次の 1 発話」の関数にする包み紙。"""

    def __init__(self, model, temperature: float = 0.0, max_new: int = 24):
        self.model, self.temperature = model, temperature
        self.max_new = max_new

    # [snippet: minigpt-call]
    @torch.no_grad()
    def __call__(self, context: str) -> str:
        self.model.eval()
        idx = torch.tensor([tok.encode(context)], device=device)
        out = self.model.generate(idx, self.max_new,  # 役割トークンも
                                  temperature=self.temperature,  # 自分で選ぶ
                                  stop_token=END_ID)
        return tok.decode(out[0, idx.size(1):].tolist())
    # [end: minigpt-call]
# [end: minigpt-llm]


# [snippet: agent-accuracy]
def agent_accuracy(llm, kind: str, n_problems: int = 100,
                   seed: int = 12345, tool=calculator) -> dict:
    """kind の問題を n_problems 問解かせ、正答率とツール使用率を返す。"""
    rng = random.Random(seed)  # 学習データ（seed 0）とは別の問題
    n_correct = n_called = 0
    for _ in range(n_problems):
        messages, answer = make_tool_example(rng, kind)
        reply, _, n_calls = run_agent(llm, messages[0]["content"], tool=tool)
        n_correct += reply == f"{answer}です。"
        n_called += n_calls > 0
    return {"acc": n_correct / n_problems, "tool_rate": n_called / n_problems}
# [end: agent-accuracy]


def direct_llm(llm):
    """ツールを使わせず、<|assistant|> から直接答えさせる版。"""
    return lambda context: "<|assistant|>" + llm(context + "<|assistant|>")


if __name__ == "__main__":
    torch.manual_seed(0)
    rows = []
    for name in ["sft", "ch23_tool_sft"]:
        llm = MiniGPTLLM(load_checkpoint(CKPT_DIR / f"{name}.pt", device))
        modes = {"ループ": (llm, calculator),
                 "直接答えさせる": (direct_llm(llm), calculator)}
        if name == "ch23_tool_sft":
            modes["電卓が 1 ずれた値を返す"] = (
                llm, lambda e: str(int(calculator(e)) + 1))
        for mode, (f, tool) in modes.items():
            for kind in ["add", "mul", "two"]:
                r = agent_accuracy(f, kind, tool=tool)
                print(f"{name:14s} {mode:12s} {kind}: 正答率 {r['acc']:5.1%}"
                      f"  ツール使用 {r['tool_rate']:5.1%}", flush=True)
    llm = MiniGPTLLM(load_checkpoint(CKPT_DIR / "ch23_tool_sft.pt", device))
    print("--- 軌跡の例 ---")
    for q in ["37×48は？", "123掛ける45は？", "25×4に7を足すといくつですか？",
              "12+35は？", "3×4×5は？", "7の2乗は？"]:
        reply, context, n = run_agent(llm, q)
        print(context)
