"""エージェントループと、台本どおりに動く偽 LLM のデモ。

実行:  python ch23/agent_loop.py  > ch23/out/agent_loop.txt
"""

from tools import TOOLS

END = "<|end|>"


# [snippet: run-agent]
def run_agent(llm, question: str, max_turns: int = 4,
              tool=TOOLS["calculator"][1], system: str = ""):
    """llm(context) は「次の 1 発話」（<|役割|>本文<|end|>）を返す関数。

    戻り値: (最終応答, 文脈の全文, ツール呼び出し回数)
    """
    context = f"<|system|>{system}{END}" if system else ""
    context += f"<|user|>{question}{END}"
    n_calls = 0
    for _ in range(max_turns):
        turn = llm(context)              # 1. 生成する
        context += turn
        if turn.startswith("<|tool|>"):  # 2. ツール呼び出しか？
            expr = turn.removeprefix("<|tool|>").removesuffix(END)
            result = tool(expr)          # 3. 外側のプログラムが実行する
            context += f"<|result|>{result}{END}"  # 4. 結果を文脈に足す
            n_calls += 1
            continue                     # 5. 続きを生成させる
        if turn.startswith("<|assistant|>"):
            answer = turn.removeprefix("<|assistant|>").removesuffix(END)
            return answer, context, n_calls  # 最終応答で終わり
        break                            # どちらでもない発話は失敗扱い
    return None, context, n_calls        # 回数の上限に達した
# [end: run-agent]


# [snippet: scripted-llm]
class ScriptedLLM:
    """台本の発話を順に返すだけの「偽 LLM」。文脈は見ない。"""

    def __init__(self, script: list[str]):
        self.script = list(script)

    def __call__(self, context: str) -> str:
        print(f"  [LLM が受け取った文脈] {context}")
        turn = self.script.pop(0)
        print(f"  [LLM の発話]           {turn}")
        return turn
# [end: scripted-llm]


if __name__ == "__main__":
    # [snippet: scripted-demo]
    llm = ScriptedLLM(["<|tool|>37*48<|end|>",
                       "<|assistant|>1776です。<|end|>"])
    answer, context, n_calls = run_agent(llm, "37×48は？")
    print("最終応答:", answer, "/ ツール呼び出し:", n_calls, "回")
    # [end: scripted-demo]
    print("--- 台本が電卓の結果を無視して適当な数を言う場合 ---")
    llm = ScriptedLLM(["<|tool|>37*48<|end|>",
                       "<|assistant|>1700です。<|end|>"])
    print("最終応答:", run_agent(llm, "37×48は？")[0])
    print("--- 台本が危険な式を渡す場合 ---")
    llm = ScriptedLLM(["<|tool|>__import__('os').system('ls')<|end|>",
                       "<|assistant|>できませんでした。<|end|>"])
    print("最終応答:", run_agent(llm, "ファイル一覧を見せて")[0])
