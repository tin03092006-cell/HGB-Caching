from __future__ import annotations
from .common import *

from .config import CacheMLConfig
from .models import make_model
from .imbalance import make_training_matrix
from .preprocess import FEATURES
from .simulator import simulate_ml_replacement


def tune_optuna(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cfg: CacheMLConfig,
    trials: int,
    cache_size: int,
    W: int,
    alpha: float,
    candidate_n: int,
    eviction_batch: int,
    Md_bound: int,
    features: list[str] | None = None,
):
    import optuna

    features = list(features or FEATURES)
    X_train, y_train, w_train, _, imbalance_report = make_training_matrix(
        train_df, features, cfg
    )

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", cfg.min_depth, cfg.max_depth)
        max_trees_by_latency = max(
            cfg.min_trees, min(cfg.max_trees, Md_bound // max(1, max_depth))
        )
        n_estimators = trial.suggest_int(
            "n_estimators", cfg.min_trees, max_trees_by_latency
        )
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100),
            "l2": trial.suggest_float("l2", 1e-10, 1.0, log=True),
        }
        model = make_model(params, cfg)
        if w_train is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=w_train)
        m = simulate_ml_replacement(
            pd.concat([train_df, valid_df], ignore_index=True),
            cache_size=cache_size,
            model=model,
            W=W,
            alpha=alpha,
            candidate_n=candidate_n,
            eviction_batch=eviction_batch,
            random_state=cfg.random_state,
            features=features,
            warmup_n=len(train_df),
        )
        return m.byte_hit_rate

    def autosave_best(study, trial):
        Path("results").mkdir(exist_ok=True)
        best = study.best_trial
        payload = {
            "best_trial": int(best.number),
            "best_value": float(best.value),
            "best_params": dict(best.params),
            "candidate_n": int(candidate_n),
            "eviction_batch": int(eviction_batch),
            "feature_count": len(features),
            "features": features,
            "imbalance": imbalance_report,
        }
        tmp = Path("results/best_params.tmp")
        out = Path("results/best_params.json")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(out)
        print(f"[autosave] best_trial={best.number} best_value={best.value:.6f}")

    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(
            objective,
            n_trials=int(trials),
            n_jobs=1,
            show_progress_bar=False,
            callbacks=[autosave_best],
        )
    except KeyboardInterrupt:
        print(
            "Stopped by user. Current best params were saved to results/best_params.json if at least one trial finished."
        )
    if not any(t.state.name == "COMPLETE" for t in study.trials):
        raise RuntimeError("No Optuna trial completed. Cannot determine best params.")
    return study.best_params, float(study.best_value)
