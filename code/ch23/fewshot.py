"""文脈内学習（few-shot）だけで、sft.pt に電卓を使わせられるか。

実行:  python ch23/fewshot.py  > ch23/out/fewshot.txt        （約 1 分）
"""

import torch

from agent_loop import run_agent
from common import CKPT_DIR, device
from mini_agent import MiniGPTLLM, agent_accuracy
from llmbook.train import load_checkpoint

torch.manual_seed(0)
# [snippet: fewshot]
SHOTS = ("<|user|>12×34は？<|end|><|tool|>12*34<|end|>"
         "<|result|>408<|end|><|assistant|>408です。<|end|>"
         "<|user|>56×7は？<|end|><|tool|>56*7<|end|>"
         "<|result|>392<|end|><|assistant|>392です。<|end|>")


class FewShotLLM(MiniGPTLLM):
    """文脈の先頭に、ツールを使うお手本の対話を 2 つ足してから生成する。"""

    def __call__(self, context: str) -> str:
        return super().__call__(SHOTS + context)
# [end: fewshot]


for name in ["base", "sft"]:
    llm = FewShotLLM(load_checkpoint(CKPT_DIR / f"{name}.pt", device))
    r = agent_accuracy(llm, "mul", n_problems=100)
    print(f"{name}.pt + お手本 2 つ: 正答率 {r['acc']:.1%}、"
          f"ツール使用 {r['tool_rate']:.1%}")
    print("  例:", run_agent(llm, "37×48は？")[1])
