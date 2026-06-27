from __future__ import annotations

class Fenwick:
   
    def __init__(self, n: int):
        self.n = int(max(1, n))
        self.bit = [0] * (self.n + 2)

    def add(self, idx0: int, delta: int) -> None:
        i = int(idx0) + 1
        n = self.n + 1
        while i <= n:
            self.bit[i] += int(delta)
            i += i & -i

    def sum_prefix(self, idx0: int) -> int:
        if idx0 < 0:
            return 0
        i = min(int(idx0) + 1, self.n + 1)
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def range_sum(self, left0: int, right0: int) -> int:
        if right0 < left0:
            return 0
        return self.sum_prefix(right0) - self.sum_prefix(left0 - 1)
