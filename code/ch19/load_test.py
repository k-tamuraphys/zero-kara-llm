"""負荷試験: 同時に質問する利用者を増やすと、速さと待ち時間はどう変わるか。

サーバ（mini_server.py）を自分で起動し、測り終えたら止める。
"""

import csv
import queue
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

from client import chat_stream
from common import make_problems

PORT = 8021
problems = make_problems(1000, seed=777)


def start_server(slots: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "ch19/mini_server.py",
                             "--port", str(PORT), "--slots", str(slots)],
                            stdout=subprocess.PIPE, text=True)
    proc.stdout.readline()  # 「待機中」の表示が出るまで待つ
    return proc


# [snippet: load-test]
def run_load(n_users: int, n_requests: int) -> dict:
    """n_users 人が、答えが返るたびに次の質問を送る。全部で n_requests 件。"""
    todo, results = queue.Queue(), []
    for p in problems[:n_requests]:
        todo.put(p)

    def user() -> None:
        while True:
            try:
                p = todo.get_nowait()
            except queue.Empty:
                return
            pieces = chat_stream(p["question"], PORT)
            reply = "".join(delta for _, delta in pieces)
            # (最初の 1 個が届くまで, 全部届くまで, 届いた回数, 正解か)
            results.append((pieces[0][0], pieces[-1][0], len(pieces),
                            reply == p["target"]))

    threads = [threading.Thread(target=user) for _ in range(n_users)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    return {"rps": n_requests / wall,
            "tps": sum(r[2] for r in results) / wall,
            "ttft": statistics.median(r[0] for r in results) * 1000,
            "total": statistics.median(r[1] for r in results) * 1000,
            "acc": sum(r[3] for r in results) / n_requests}
# [end: load-test]


if __name__ == "__main__":
    rows = []
    for slots in [1, 16]:
        server = start_server(slots)
        run_load(2, 50)  # 準備運動
        print(f"スロット {slots} 個のサーバ")
        for n_users in [1, 2, 4, 8, 16, 32]:
            # 3 回測り、ほかの処理にいちばん邪魔されなかった回を採る
            r = max((run_load(n_users, 200) for _ in range(3)),
                    key=lambda r: r["rps"])
            rows.append((slots, n_users, r))
            print(f"  同時 {n_users:2d} 人: {r['rps']:5.0f} 件/秒"
                  f"  {r['tps']:5.0f} トークン/秒"
                  f"  最初の 1 個まで {r['ttft']:6.1f} ミリ秒"
                  f"  応答完了まで {r['total']:6.1f} ミリ秒")
        if slots == 16:
            r = run_load(16, 1000)
            print(f"  1,000 問すべてを同時 16 人で送ったときの正答率: "
                  f"{r['acc']:.1%}")
        server.terminate()
        server.wait()
    with open("ch19/out/load_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slots", "users", "rps", "tps", "ttft_ms", "total_ms"])
        for slots, n_users, r in rows:
            w.writerow([slots, n_users, f"{r['rps']:.1f}", f"{r['tps']:.0f}",
                        f"{r['ttft']:.2f}", f"{r['total']:.2f}"])
