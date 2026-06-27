from __future__ import annotations
from .common import *
from scipy.stats import boxcox

from .config import CacheMLConfig
from .system_power import available_ram_budget_bytes
from .feature_utils import Fenwick

FEATURES = [
    "size",
    "log_size",
    "type_code",
    "op_code",
    "ttl",
    "log_ttl",
    "recency",
    "log_recency",
    "count_w",
    "count_short",
    "count_mid",
    "count_long",
    "ewma_freq",
    "log_freq_seen",
    "freq_density",
    "last_gap",
    "prev_gap",
    "gap_ewma",
    "gap_mean",
    "gap_std",
    "gap_cv",
    "burst_ratio",
    "acceleration",
    "unique_since_last",
    "log_unique_since_last",
    "byte_freq",
    "byte_ewma",
    "byte_count_w",
    "age_seen",
    "log_age_seen",
]
INF_NAT = 10**12
_FEAT_INDEX = {f: i for i, f in enumerate(FEATURES)}


def next_access_distance(obj: pd.Series) -> np.ndarray:
    n, objs, nxt, y = len(obj), obj.astype(str).to_numpy(), {}, np.empty(len(obj), dtype=np.float64)
    for i in range(n - 1, -1, -1):
        y[i] = float(nxt.get(objs[i], n + 1) - i)
        nxt[objs[i]] = i
    return y


def choose_W_acf(obj: pd.Series, threshold: float = 0.05, max_lag: int = 4096) -> int:
    n = len(obj)
    if n < 10:
        return 1
    x = pd.Categorical(obj).codes.astype(np.float64)
    x -= x.mean()
    denom, max_lag = float(np.dot(x, x)) or 1.0, max(1, min(int(max_lag), n - 2))
    for k in range(1, max_lag + 1):
        if abs(float(np.dot(x[:-k], x[k:]) / denom)) <= threshold:
            return k
    return max_lag


def choose_alpha_mle(df: pd.DataFrame, top_hot: int = 50) -> float:
    gaps, arr = [], df["obj"].to_numpy()
    for o in df["obj"].value_counts().head(max(1, top_hot)).index:
        if len(pos := np.flatnonzero(arr == o)) > 1:
            gaps.extend(np.diff(pos).tolist())
    return float(1.0 - np.exp(-1.0 / max(float(np.mean(gaps)) if gaps else 0.1, 1.0)))


def estimate_record_bytes(df: pd.DataFrame) -> float:
    return float(
        df[
            [
                c
                for c in ["obj", "time_s", "t", "size", "type", "op", "ttl"]
                if c in df.columns
            ]
        ]
        .memory_usage(deep=True)
        .sum()
        / max(1, len(df))
    )


def _window_scales(W: int) -> tuple[int, int, int]:
    return (
        max(1, (W := max(1, int(W))) // 16),
        max(max(1, W // 16), W // 4),
        max(max(max(1, W // 16), W // 4), W),
    )


class FeatureState:
    def __init__(self, W: int, alpha: float, df_len: int):
        self.W, self.alpha = max(1, int(W)), alpha
        self.Ws, self.Wm, self.Wl = _window_scales(W)
        self.last, self.first, self.meta = {}, {}, {}
        self.gn, self.gm, self.gm2, self.gew, self.pgap = (
            defaultdict(int),
            defaultdict(float),
            defaultdict(float),
            defaultdict(float),
            defaultdict(float),
        )
        self.hs, self.hm, self.hl = (
            defaultdict(deque),
            defaultdict(deque),
            defaultdict(deque),
        )
        self.ewma, self.seen = defaultdict(float), defaultdict(int)
        self.bit = Fenwick(df_len + 2)

    def clean(self, o: str, i: int):
        for h, w in [
            (self.hs[o], self.Ws),
            (self.hm[o], self.Wm),
            (self.hl[o], self.Wl),
        ]:
            while h and i - h[0] > w:
                h.popleft()

    def update(self, o: str, i: int, sz: float, tc: float, oc: float, ttl: float):
        self.meta[o] = (sz, tc, oc, ttl)
        self.clean(o, i)
        if o not in self.last:
            self.first[o] = i
        else:
            gap = float(i - self.last[o])
            self.pgap[o] = gap
            self.gn[o] += 1
            gn = self.gn[o]
            d = gap - self.gm[o]
            self.gm[o] += d / gn
            self.gm2[o] += d * (gap - self.gm[o])
            self.gew[o] = (
                gap if gn == 1 else (1 - self.alpha) * self.gew[o] + self.alpha * gap
            )
            self.bit.add(self.last[o], -1)
        self.last[o] = i
        self.bit.add(i, 1)
        self.hs[o].append(i)
        self.hm[o].append(i)
        self.hl[o].append(i)
        self.ewma[o] = (1 - self.alpha) * self.ewma[o] + self.alpha
        self.seen[o] += 1

    def get_features(self, o: str, i: int, features: list[str]) -> tuple | list:
        sz, tc, oc, ttl = self.meta.get(o, (1.0, 0.0, 0.0, 0.0))
        self.clean(o, i)
        has_last, seen, age = (
            o in self.last,
            self.seen[o],
            float(i - self.first.get(o, i)),
        )
        rec = float(i - self.last[o]) if has_last else float(self.Wl + 1)
        unq = (
            float(self.bit.range_sum(self.last[o] + 1, i - 1))
            if has_last
            else float(len(self.last))
        )
        gn, gm = self.gn[o], self.gm[o] if self.gn[o] > 0 else rec
        gs = math.sqrt(max(self.gm2[o] / max(1, gn - 1), 0.0)) if gn > 1 else 0.0
        cs, cm, cl = (
            float(len(self.hs[o])),
            float(len(self.hm[o])),
            float(len(self.hl[o])),
        )
        ew, prev, gew = (
            self.ewma[o],
            self.pgap[o] if has_last else float(self.Wl + 1),
            self.gew[o] if gn > 0 else rec,
        )
        vals = (
            sz, math.log1p(sz), tc, oc, ttl, math.log1p(max(0.0, ttl)),
            rec, math.log1p(rec), cl, cs, cm, cl, ew, math.log1p(seen),
            seen / (age + 1.0), rec, prev, gew, gm, gs, gs / (gm + 1.0),
            cs / (cl + 1.0), cs - cm * (self.Ws / max(1.0, float(self.Wm))),
            unq, math.log1p(unq), sz * seen, sz * ew, sz * cl,
            age, math.log1p(age),
        )
        if features is FEATURES:
            return vals
        return [vals[_FEAT_INDEX[f]] for f in features]


def build_features(df: pd.DataFrame, W: int, alpha: float) -> pd.DataFrame:
    n, state, out = (
        len(df),
        FeatureState(W, alpha, len(df)),
        {f: np.empty(len(df), dtype=np.float64) for f in FEATURES},
    )
    objs, sizes = df["obj"].astype(str).to_numpy(), df["size"].astype(float).to_numpy()
    tcs = pd.Categorical(df["type"].astype(str)).codes.astype(np.float64)
    ocs = pd.Categorical(df["op"].astype(str)).codes.astype(np.float64)
    ttls = (
        pd.to_numeric(df["ttl"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )

    for i in range(n):
        o, sz, tc, oc, ttl = objs[i], sizes[i], tcs[i], ocs[i], ttls[i]
        vals = state.get_features(o, i, FEATURES)
        for j, f in enumerate(FEATURES):
            out[f][i] = vals[j]
        state.update(o, i, sz, tc, oc, ttl)

    res = df.copy()
    for f in FEATURES:
        res[f] = out[f]
    res["size"] = sizes
    return res


def prepare(df: pd.DataFrame, cfg: CacheMLConfig) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        raise ValueError("Trace is empty: CSV has 0 request rows.")
    W_acf = choose_W_acf(df["obj"], cfg.acf_threshold, cfg.acf_max_lag)
    record_bytes = estimate_record_bytes(df)
    ram_budget = available_ram_budget_bytes(cfg)
    W, alpha = max(
        1, min(W_acf, int(ram_budget / max(record_bytes, 1.0)))
    ), choose_alpha_mle(df, cfg.top_hot_objects)
    y_nat = next_access_distance(df["obj"])
    y_bc, lam = (
        (np.log1p(y_nat), 0.0) if np.allclose(y_nat, y_nat[0]) else boxcox(y_nat)
    )
    out = build_features(df, W, alpha)
    out["nat"], out["y"] = y_nat, y_bc.astype(np.float64)
    W_short, W_mid, W_long = _window_scales(W)
    return out, {
        "lambda": float(lam),
        "W": int(W),
        "W_short": int(W_short),
        "W_mid": int(W_mid),
        "W_long": int(W_long),
        "W_acf": int(W_acf),
        "W_ram_max_records": int(ram_budget / max(record_bytes, 1.0)),
        "record_bytes": float(record_bytes),
        "feature_ram_budget_bytes": int(ram_budget),
        "alpha": float(alpha),
        "features": FEATURES,
        "feature_version": "v2_locality_burst_reuse_byte_ttl",
    }
