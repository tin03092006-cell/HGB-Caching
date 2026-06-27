from __future__ import annotations
from .common import *


def load_trace(path: str, max_rows: int = 0) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    nrows = None if not max_rows or max_rows <= 0 else int(max_rows)
    # The dataset is strictly CSV with obj,size,type,timestamp,op,ttl
    df = pd.read_csv(p, nrows=nrows)

    out = pd.DataFrame()
    out["obj"] = df["obj"].astype(str)
    out["size"] = df["size"].astype(float)
    out["type"] = df["type"].astype(str)
    out["op"] = df["op"].astype(str)
    out["ttl"] = df["ttl"].astype(float)
    out["time_s"] = df["timestamp"].astype(float)
    out["t"] = np.arange(len(out), dtype=np.int64)

    return out
