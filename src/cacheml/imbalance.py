from __future__ import annotations
from .common import *


def describe_imbalance(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "obj" not in df.columns:
        return {"rows": len(df), "objects": 0}
    c = df["obj"].value_counts()
    return {
        "rows": len(df),
        "objects": len(c),
        "max_freq": int(c.max()),
        "median_freq": float(c.median()),
        "mean_freq": float(c.mean()),
        "one_hit_objects": int((c == 1).sum()),
        "one_hit_object_ratio": float((c == 1).sum() / max(1, len(c))),
        "top_1pct_request_share": float(
            c.iloc[: max(1, int(np.ceil(0.01 * len(c))))].sum() / len(df)
        ),
        "top_5pct_request_share": float(
            c.iloc[: max(1, int(np.ceil(0.05 * len(c))))].sum() / len(df)
        ),
    }


def downsample_frequent_objects(
    train_df: pd.DataFrame, max_per_object: int = 0, random_state: int = 42
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rep = {
        "enabled": bool(max_per_object > 0),
        "max_per_object": max_per_object or 0,
        "rows_before": len(train_df),
        "rows_after": len(train_df),
    }
    if not max_per_object or max_per_object <= 0 or train_df.empty:
        return train_df, rep
    out = (
        train_df.sample(frac=1, random_state=random_state)
        .groupby("obj", sort=False)
        .head(max_per_object)
        .sort_index()
    )
    rep.update(
        {
            "rows_after": len(out),
            "removed_rows": len(train_df) - len(out),
            "capped_objects": int(
                (train_df["obj"].value_counts() > max_per_object).sum()
            ),
        }
    )
    return out, rep


def make_sample_weights(
    train_df: pd.DataFrame,
    strategy: str = "hybrid",
    hot_strength: float = 0.50,
    rare_strength: float = 0.75,
    size_strength: float = 0.25,
    max_weight: float = 20.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    st = (strategy or "hybrid").lower()
    if st == "none" or train_df.empty:
        return None, {"strategy": st, "enabled": False}
    freq = train_df["obj"].map(train_df["obj"].value_counts()).astype(float).to_numpy()
    mf = max(1.0, freq.max())

    if st == "hot":
        w = 1.0 + np.log1p(freq)
    elif st == "rare":
        w = 1.0 + np.log1p(mf / np.maximum(freq, 1.0))
    elif st == "balanced":
        w = 1.0 / np.sqrt(np.maximum(freq, 1.0))
        w /= max(float(np.mean(w)), 1e-12)
    elif st == "hybrid":
        sz = (
            np.log1p(train_df["size"].fillna(1.0).astype(float))
            if "size" in train_df
            else np.zeros_like(freq)
        )
        w = (
            1.0
            + hot_strength * np.log1p(freq)
            + rare_strength * np.log1p(mf / np.maximum(freq, 1.0))
            + size_strength * (sz / max(float(np.median(sz)), 1e-12))
        )
    else:
        raise ValueError(f"Unknown strategy: {st}")

    if max_weight > 0:
        w = np.minimum(w, max_weight)
    if (m := float(np.mean(w))) > 0:
        w /= m
    return w.astype(float), {
        "strategy": st,
        "enabled": True,
        "hot_strength": hot_strength,
        "rare_strength": rare_strength,
        "size_strength": size_strength,
        "max_weight": max_weight,
        "weight_min": float(np.min(w)),
        "weight_mean": float(np.mean(w)),
        "weight_max": float(np.max(w)),
    }


def make_training_matrix(
    train_df: pd.DataFrame, features: list[str], cfg
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame, dict[str, Any]]:
    b = describe_imbalance(train_df)
    df, dr = downsample_frequent_objects(
        train_df,
        getattr(cfg, "downsample_max_per_object", 0),
        getattr(cfg, "random_state", 42),
    )
    w, wr = make_sample_weights(
        df,
        getattr(cfg, "imbalance_strategy", "hybrid"),
        getattr(cfg, "hot_weight_strength", 0.5),
        getattr(cfg, "rare_weight_strength", 0.75),
        getattr(cfg, "size_weight_strength", 0.25),
        getattr(cfg, "sample_weight_max", 20.0),
    )
    return (
        df[features].to_numpy(dtype=float),
        df["y"].to_numpy(dtype=float),
        w,
        df,
        {
            "before": b,
            "after": describe_imbalance(df),
            "downsampling": dr,
            "sample_weights": wr,
        },
    )


def save_imbalance_report(report: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
