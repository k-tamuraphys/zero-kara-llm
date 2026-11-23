"""最小の推論サーバ。標準ライブラリだけで /v1/chat/completions を提供する。

実行:  python ch19/mini_server.py [--port 8019] [--slots 16] [--model sft]
"""

import argparse
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from batcher import Batcher, Job
from common import load
from engine import Engine
from llmbook.chat import render_chat

inbox: queue.Queue = queue.Queue()  # 受付スレッド → エンジンスレッド


# [snippet: engine-loop]
def engine_loop(batcher: Batcher) -> None:
    """モデルを動かすのは、このスレッドだけ。連続バッチ処理で回し続ける。"""
    while True:
        jobs = []
        if not batcher.running:
            jobs.append(inbox.get())  # 仕事がなければ、来るまで眠る
        while len(jobs) < len(batcher.free) and not inbox.empty():
            jobs.append(inbox.get())  # 空きスロットの数だけ、待ち行列から取る
        if jobs:
            batcher.admit(jobs)       # まとめてプロンプトを読ませる
        if batcher.running:
            batcher.step()            # 全員を 1 トークン進める
# [end: engine-loop]


class Handler(BaseHTTPRequestHandler):
    # [snippet: do-post]
    def do_POST(self):
        """リクエスト 1 件ごとに、専用のスレッドでこの関数が呼ばれる。"""
        if self.path != "/v1/chat/completions":
            return self.send_error(404)
        size = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(size))
        prompt = tok.encode(render_chat(body["messages"],
                                        add_generation_prompt=True))
        if len(prompt) >= model.config.block_size:
            return self.send_error(400, "prompt too long")
        events = queue.Queue()  # エンジンスレッド → このリクエスト
        job = Job(prompt, body.get("max_tokens", 24),
                  body.get("temperature", 0.0),
                  on_token=lambda token, done: events.put((token, done)))
        inbox.put(job)          # エンジンに仕事を渡す
        if body.get("stream"):
            self.reply_stream(events)
        else:
            self.reply_whole(events, len(prompt))
    # [end: do-post]

    # [snippet: reply-stream]
    def reply_stream(self, events: queue.Queue) -> None:
        """トークンが出るたびに、その場で 1 行ずつ送る。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        tokens, n_sent, done = [], 0, False
        # [snippet: stream-loop]
        while not done:
            token, done = events.get()  # 次のトークンが出るまで待つ
            if token != END_ID:
                tokens.append(token)
            text = tok.decode(tokens)
            if text.endswith("\ufffd") and not done:
                continue  # 文字の途中で切れている。次のトークンを待つ
            chunk = {"choices": [{"index": 0,
                                  "delta": {"content": text[n_sent:]},
                                  "finish_reason": "stop" if done else None}]}
            line = "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
            self.wfile.write(line.encode())
            self.wfile.flush()          # ためずに、すぐ送り出す
            n_sent = len(text)
        # [end: stream-loop]
        self.wfile.write(b"data: [DONE]\n\n")
    # [end: reply-stream]

    def reply_whole(self, events: queue.Queue, n_prompt: int) -> None:
        """最後まで生成してから、まとめて 1 回で返す。"""
        tokens, done = [], False
        while not done:
            token, done = events.get()
            tokens.append(token)
        text = tok.decode(tokens).split("<|end|>")[0]
        reply = {"object": "chat.completion",
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant",
                                          "content": text}}],
                 "usage": {"prompt_tokens": n_prompt,
                           "completion_tokens": len(tokens),
                           "total_tokens": n_prompt + len(tokens)}}
        data = json.dumps(reply, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass  # アクセスログを画面に出さない


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--model", default="sft")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, tok = load(args.model)
    END_ID = tok.special_tokens["<|end|>"]
    # [snippet: server-main]
    batcher = Batcher(Engine(model, n_slots=args.slots), stop_id=END_ID)
    threading.Thread(target=engine_loop, args=(batcher,), daemon=True).start()
    # 一度に届いた接続を待たせておける数（既定の 5 では、100 人が同時に
    # つなぐと断られる接続が出る）
    ThreadingHTTPServer.request_queue_size = 128
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"http://127.0.0.1:{args.port}/v1/chat/completions で待機中",
          flush=True)
    server.serve_forever()
    # [end: server-main]
