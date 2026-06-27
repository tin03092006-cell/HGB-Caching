from __future__ import annotations
from .common import *
from .config import CacheMLConfig


def set_full_cpu_threads(workers: int) -> int:
    workers = max(1, int(workers or os.cpu_count() or 1))
    for k in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[k] = str(workers)
    return workers


def available_ram_budget_bytes(cfg: CacheMLConfig) -> int:
    v = psutil.virtual_memory().available
    return max(
        64 * 1024 * 1024,
        min(
            int(v * float(cfg.ram_fraction)),
            max(64 * 1024 * 1024, int(v) - int(cfg.reserve_ram_mb * 1024 * 1024)),
        ),
    )


def infer_request_rate(df) -> tuple[float, float]:
    t = df["time_s"].to_numpy(dtype=float)[
        np.isfinite(df["time_s"].to_numpy(dtype=float))
    ]
    if len(t) < 2:
        return float(len(df)), float(len(df))
    return float(len(t) / max(float(t[-1] - t[0]), 1e-9)), float(
        max(1, np.bincount(np.floor(t - t[0]).astype(np.int64)).max(initial=1))
    )


def measure_predict_us_per_object(model, X, repeats: int = 5) -> float:
    if X is None or len(X) == 0:
        return 1.0
    Xb = X[: min(len(X), 4096)]
    try:
        model.predict(Xb[:32])
    except Exception:
        return 1.0

    def _time():
        t0 = time.perf_counter()
        model.predict(Xb)
        return (time.perf_counter() - t0) * 1e6 / max(len(Xb), 1)

    return float(max(min([_time() for _ in range(max(1, repeats))]), 1e-6))


def measure_node_us(model, X) -> float:
    return float(
        measure_predict_us_per_object(model, X)
        / max(
            1,
            int(getattr(model, "n_estimators", getattr(model, "max_iter", 100)) or 100)
            * int(getattr(model, "max_depth", 8) or 8),
        )
    )


def build_hardware_plan(
    cfg: CacheMLConfig,
    df,
    model=None,
    X_probe=None,
    W_acf: int | None = None,
    record_bytes: float | None = None,
) -> dict[str, Any]:
    w, rb, (R, Rp) = (
        set_full_cpu_threads(cfg.n_workers),
        available_ram_budget_bytes(cfg),
        infer_request_rate(df),
    )
    record_bytes = float(
        record_bytes
        if record_bytes is not None
        else df[[c for c in ["obj", "time_s", "t", "size", "type"] if c in df.columns]]
        .memory_usage(deep=True)
        .sum()
        / max(1, len(df))
    )
    W_ram, predict_us, node_us = int(rb / max(record_bytes, 1.0)), 1.0, 0.01
    if model is not None and X_probe is not None:
        predict_us, node_us = measure_predict_us_per_object(
            model, X_probe
        ), measure_node_us(model, X_probe)
    N = (
        int(cfg.candidate_n)
        if cfg.candidate_n and cfg.candidate_n > 0
        else max(1, int(cfg.latency_budget_us / max(predict_us, 1e-9)))
    )
    B = (
        int(cfg.eviction_batch)
        if cfg.eviction_batch and cfg.eviction_batch > 0
        else max(1, int(math.ceil(R * (N * predict_us) / 1e6)))
    )
    return {
        "workers": w,
        "ram_available_bytes": int(psutil.virtual_memory().available),
        "ram_budget_bytes": rb,
        "ram_fraction": float(cfg.ram_fraction),
        "reserve_ram_mb": float(cfg.reserve_ram_mb),
        "record_bytes": record_bytes,
        "R_req_per_s": float(R),
        "R_peak_req_per_s": float(Rp),
        "W_acf": int(W_acf) if W_acf is not None else None,
        "W_ram_max_records": W_ram,
        "W_final": int(max(1, min(W_acf, W_ram) if W_acf else W_ram)),
        "predict_us_per_object": float(predict_us),
        "cpu_c_us_per_tree_node": float(node_us),
        "latency_budget_us": float(cfg.latency_budget_us),
        "N_max": max(1, int(cfg.latency_budget_us / max(predict_us, 1e-9))),
        "p_from_hardware": float(
            cfg.p_from_nmax(
                max(1, int(cfg.latency_budget_us / max(predict_us, 1e-9))),
                cfg.success_prob,
            )
        ),
        "candidate_n": max(1, N),
        "eviction_batch": max(1, B),
        "model_product_Md_bound": int(
            max(1, int(cfg.latency_budget_us / max(N * node_us, 1e-9)))
        ),
        "config": asdict(cfg),
    }


def save_plan(plan: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
