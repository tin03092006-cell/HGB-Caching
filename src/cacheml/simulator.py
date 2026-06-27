from __future__ import annotations
from .common import *


from .preprocess import FEATURES, INF_NAT, FeatureState


@dataclass
class Metrics:
    policy: str
    requests: int
    hits: int
    misses: int
    hit_rate: float
    byte_hit_rate: float
    avg_latency_us: float = 0.0
    candidate_n: int = 0
    eviction_batch: int = 0
    ai_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _base_metrics(
    policy: str,
    n: int,
    hits: int,
    byte_hits: float,
    total_bytes: float,
    latency_sum_us: float = 0.0,
    ai_calls: int = 0,
    candidate_n: int = 0,
    eviction_batch: int = 0,
) -> Metrics:
    return Metrics(
        policy,
        n,
        hits,
        n - hits,
        float(hits / max(1, n)),
        float(byte_hits / max(total_bytes, 1.0)),
        float(latency_sum_us / max(1, ai_calls)) if ai_calls else 0.0,
        int(candidate_n or 0),
        int(eviction_batch or 0),
        int(ai_calls),
    )


def simulate_lru(df: pd.DataFrame, cache_size: int) -> Metrics:
    cache, hits, byte_hits = OrderedDict(), 0, 0.0
    for o, sz in zip(df["obj"].astype(str), df["size"].astype(float)):
        if o in cache:
            hits += 1
            byte_hits += sz
            cache.move_to_end(o)
        elif cache_size > 0:
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[o] = None
    return _base_metrics("LRU", len(df), hits, byte_hits, df["size"].sum())


def simulate_lfu(df: pd.DataFrame, cache_size: int) -> Metrics:
    cache, freq, ver, heap, hits, byte_hits = (
        set(),
        defaultdict(int),
        defaultdict(int),
        [],
        0,
        0.0,
    )
    for i, (o, sz) in enumerate(zip(df["obj"].astype(str), df["size"].astype(float))):
        if o in cache:
            hits += 1
            byte_hits += sz
        elif cache_size > 0:
            if len(cache) >= cache_size:
                while heap:
                    f, _, v, c = heapq.heappop(heap)
                    if c in cache and v == ver[c] and f == freq[c]:
                        cache.remove(c)
                        break
            cache.add(o)
        freq[o] += 1
        ver[o] += 1
        heapq.heappush(heap, (freq[o], i, ver[o], o))
    return _base_metrics("LFU", len(df), hits, byte_hits, df["size"].sum())


def simulate_belady(df: pd.DataFrame, cache_size: int) -> Metrics:
    cache, ver, heap, hits, byte_hits = {}, defaultdict(int), [], 0, 0.0
    nat = (
        df["nat"].to_numpy(dtype=float)
        if "nat" in df.columns
        else np.full(len(df), len(df) + 1, dtype=float)
    )
    next_idx = np.where((idx := np.arange(len(df)) + nat) >= len(df), INF_NAT, idx)
    for i, (o, sz) in enumerate(zip(df["obj"].astype(str), df["size"].astype(float))):
        nx = next_idx[i]
        if o in cache:
            hits += 1
            byte_hits += sz
        elif cache_size > 0:
            if len(cache) >= cache_size:
                while heap:
                    neg_nx, v, c = heapq.heappop(heap)
                    if c in cache and cache[c][1] == v and cache[c][0] == -neg_nx:
                        del cache[c]
                        break
        if cache_size > 0 or o in cache:
            ver[o] += 1
            cache[o] = (nx, ver[o])
            heapq.heappush(heap, (-float(nx), ver[o], o))
    return _base_metrics("Belady", len(df), hits, byte_hits, df["size"].sum())


class CacheSet:
    def __init__(self):
        self.items, self.pos = [], {}

    def __contains__(self, x):
        return x in self.pos

    def __len__(self):
        return len(self.items)

    def add(self, x):
        if x not in self.pos:
            self.pos[x] = len(self.items)
            self.items.append(x)

    def remove(self, x):
        if x in self.pos:
            j, last = self.pos.pop(x), self.items.pop()
            if j < len(self.items):
                self.items[j], self.pos[last] = last, j


def simulate_ml_replacement(
    df: pd.DataFrame,
    cache_size: int,
    model,
    W: int,
    alpha: float,
    candidate_n: int,
    eviction_batch: int,
    random_state: int = 42,
    features: list[str] | None = None,
) -> Metrics:
    features = list(features or FEATURES)
    cache, victim_queue, state = CacheSet(), deque(), FeatureState(W, alpha, len(df))
    objs, sizes = df["obj"].astype(str).to_numpy(), df["size"].astype(float).to_numpy()
    tcs = (
        df["type_code"].to_numpy(dtype=float)
        if "type_code" in df.columns
        else np.zeros(len(df))
    )
    ocs = (
        df["op_code"].to_numpy(dtype=float)
        if "op_code" in df.columns
        else np.zeros(len(df))
    )
    ttls = df["ttl"].to_numpy(dtype=float) if "ttl" in df.columns else np.zeros(len(df))
    hits, byte_hits, latency, ai_calls, rng = (
        0,
        0.0,
        0.0,
        0,
        np.random.default_rng(random_state),
    )

    def choose_victim(i: int, avoid: str) -> str | None:
        nonlocal latency, ai_calls
        while victim_queue:
            if (v := victim_queue.popleft()) in cache and v != avoid:
                return v
        if not cache:
            return None
        cands = [
            cache.items[j]
            for j in (
                rng.choice(
                    len(cache.items), min(candidate_n, len(cache)), replace=False
                ).tolist()
                if candidate_n < len(cache)
                else range(len(cache.items))
            )
            if cache.items[j] != avoid
        ] or [cache.items[0]]
        t0 = time.perf_counter()
        pred = model.predict(
            np.asarray([state.get_features(c, i, features) for c in cands], dtype=float)
        )
        latency += (time.perf_counter() - t0) * 1e6
        ai_calls += 1
        order = np.argsort(-np.asarray(pred, dtype=float))
        victim_queue.extend(cands[j] for j in order[:eviction_batch])
        while victim_queue:
            if (v := victim_queue.popleft()) in cache and v != avoid:
                return v
        return cands[order[0]]

    for i, (o, sz, tc, oc, ttl) in enumerate(zip(objs, sizes, tcs, ocs, ttls)):
        state.meta[o] = (float(sz), float(tc), float(oc), float(ttl))
        if o in cache:
            hits += 1
            byte_hits += float(sz)
        else:
            if len(cache) >= cache_size > 0 and (v := choose_victim(i, o)) in cache:
                cache.remove(v)
            if cache_size > 0:
                cache.add(o)
        state.update(o, i, float(sz), float(tc), float(oc), float(ttl))

    return _base_metrics(
        "ML",
        len(df),
        hits,
        byte_hits,
        float(sizes.sum()),
        latency,
        ai_calls,
        candidate_n,
        eviction_batch,
    )
