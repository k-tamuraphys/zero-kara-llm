"""推論サーバに質問を送り、応答が届く時刻を測る。

先に別の端末で  python ch19/mini_server.py  を動かしておく。
"""

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:{port}/v1/chat/completions"


# [snippet: client]
def chat_stream(question: str, port: int = 8019, max_tokens: int = 24):
    """質問を送り、(届いた時刻 [秒], 届いた文字列) のリストを返す。"""
    body = {"messages": [{"role": "user", "content": question}],
            "stream": True, "max_tokens": max_tokens}
    request = urllib.request.Request(
        URL.format(port=port), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    pieces, t0 = [], time.perf_counter()
    with urllib.request.urlopen(request) as response:
        for line in response:  # サーバが 1 行送るたびに、ここが 1 回まわる
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"]["content"]
            pieces.append((time.perf_counter() - t0, delta))
    return pieces
# [end: client]


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8019
    chat_stream("1+1は？", port)  # 1 回目は準備運動（接続の初期化など）
    for question in sys.argv[2:] or ["12+35は？"]:
        pieces = chat_stream(question, port)
        print(f"質問: {question}")
        for t, delta in pieces:
            print(f"  {t * 1000:6.1f} ミリ秒  {delta!r}")
        print(f"  応答: {''.join(d for _, d in pieces)}")
