"""連続バッチ処理: 空いたスロットに、待っているリクエストをすぐ入れる。"""

import torch

from engine import Engine, pick


class Job:
    """1 件の生成リクエスト。"""

    def __init__(self, prompt: list[int], max_tokens: int = 24,
                 temperature: float = 0.0, on_token=None):
        self.prompt, self.max_tokens = prompt, max_tokens
        self.temperature = temperature
        self.on_token = on_token  # トークンが出るたびに呼ばれる関数
        self.tokens: list[int] = []


# [snippet: batcher]
class Batcher:
    def __init__(self, engine: Engine, stop_id: int):
        self.engine, self.stop_id = engine, stop_id
        self.free = list(range(engine.n_slots))  # 空いているスロット
        self.running: dict[int, Job] = {}        # スロット番号 -> 実行中の仕事
        self.last = torch.zeros(engine.n_slots, dtype=torch.long)

    def admit(self, jobs: list[Job]) -> None:
        """空きスロットに仕事を入れ、プロンプトを読ませて 1 個目を出す。"""
        slots = [self.free.pop() for _ in jobs]
        self.running.update(zip(slots, jobs))
        logits = self.engine.prefill(torch.tensor(slots),
                                     [job.prompt for job in jobs])
        for i, (slot, job) in enumerate(zip(slots, jobs)):
            self.emit(slot, int(pick(logits[i:i + 1], job.temperature)))

    def emit(self, slot: int, token: int) -> None:
        """スロットの仕事にトークンを 1 個渡す。終わりならスロットを空ける。"""
        job = self.running[slot]
        job.tokens.append(token)
        self.last[slot] = token
        done = (token == self.stop_id or len(job.tokens) >= job.max_tokens
                or self.engine.length[slot] >= self.engine.max_len)
        if job.on_token is not None:
            job.on_token(token, done)
        if done:
            del self.running[slot]
            self.free.append(slot)

    def step(self) -> None:
        """実行中のすべての仕事を、1 回の計算で 1 トークンずつ進める。"""
        slots = torch.tensor(sorted(self.running))
        logits = self.engine.forward(slots, self.last[slots][:, None])
        for i, slot in enumerate(slots.tolist()):
            job = self.running[slot]
            self.emit(slot, int(pick(logits[i:i + 1], job.temperature)))
# [end: batcher]
