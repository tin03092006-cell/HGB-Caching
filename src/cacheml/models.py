from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .config import CacheMLConfig
from .preprocess import FEATURES
from .system_power import set_full_cpu_threads
from .imbalance import make_sample_weights, make_training_matrix

def split_train_valid_test(df: pd.DataFrame):
    n = len(df)
    n_test = int(round(n * 0.15))
    n_valid = int(round(n * 0.15))
    n_train = max(1, n - n_valid - n_test)
    train = df.iloc[:n_train].copy()
    valid = df.iloc[n_train : n_train + n_valid].copy()
    test = df.iloc[n_train + n_valid :].copy()
    return train, valid, test

def make_model(params: dict, cfg: CacheMLConfig):
    from sklearn.ensemble import HistGradientBoostingRegressor
    
    set_full_cpu_threads(cfg.n_workers)
    return HistGradientBoostingRegressor(
        max_iter=int(params.get("n_estimators", 256)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_leaf_nodes=int(params.get("num_leaves", 31)),
        max_depth=int(params.get("max_depth", 8)),
        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
        l2_regularization=float(params.get("l2", 0.0)),
        early_stopping=True,
        random_state=cfg.random_state,
    )

def default_params(cfg: CacheMLConfig, Md_bound: int | None = None) -> dict:
    max_depth = min(cfg.max_depth, 8)
    if Md_bound:
        n_estimators = max(cfg.min_trees, min(cfg.max_trees, int(Md_bound // max(1, max_depth))))
    else:
        n_estimators = min(cfg.max_trees, 256)
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_samples_leaf": 20,
    }

def train_model(
    train_df: pd.DataFrame,
    cfg: CacheMLConfig,
    params: dict | None = None,
    features: list[str] | None = None,
):
    params = params or default_params(cfg)
    features = list(features or FEATURES)
    model = make_model(params, cfg)
    X, y, w, _, _ = make_training_matrix(train_df, features, cfg)
    if w is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=w)
    return model, params

def save_model(model, params: dict, path: str, features: list[str] | None = None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "params": params, "features": list(features or FEATURES)}, p)


