"""負荷試験の続き: 100 人が同時に使ったら。1,000 問をサーバ経由で解くと何秒か。

load_test.py とは別の機会に測るので、比べる基準（スロット 16 個・同時 32 人）
も同じ実行の中で測り直す。
"""

import threading

import load_test
from load_test import run_load, start_server

load_test.PORT = 8022
failures = []  # 途中で失敗した利用者（接続を断られた、など）
threading.excepthook = lambda args: failures.append(args.exc_type.__name__)


def best(n_users: int, n_requests: int, repeat: int = 3) -> dict:
    """repeat 回測り、ほかの処理にいちばん邪魔されなかった回を採る。"""
    return max((run_load(n_users, n_requests) for _ in range(repeat)),
               key=lambda r: r["rps"])


def show(n_users: int, r: dict) -> None:
    print(f"  同時 {n_users:3d} 人: {r['rps']:5.0f} 件/秒"
          f"  最初の 1 個まで {r['ttft']:7.1f} ミリ秒"
          f"  応答完了まで {r['total']:7.1f} ミリ秒（どちらも中央値）")


if __name__ == "__main__":
    for slots in [1, 16, 64]:
        server = start_server(slots)
        run_load(2, 50)  # 準備運動
        print(f"スロット {slots} 個のサーバ")
        if slots == 16:
            show(32, best(32, 200))
        show(100, best(100, 400))
        if slots == 16:
            r = best(32, 1000, repeat=2)
            print(f"  1,000 問を同時 32 人で送る: {1000 / r['rps']:.1f} 秒"
                  f"（{r['rps']:.0f} 件/秒）  正答率 {r['acc']:.1%}")
        server.terminate()
        server.wait()
    print(f"失敗した質問: {len(failures)} 件")
