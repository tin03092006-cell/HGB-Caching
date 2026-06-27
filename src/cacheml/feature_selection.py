from __future__ import annotations
from .common import *
from sklearn.inspection import permutation_importance


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.maximum(
        np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    return x / s if (s := float(x.sum())) > 0 else x


def _tree_importance(model: Any, features: list[str]) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        if (imp := getattr(model, "feature_importances_")) is not None and np.asarray(
            imp
        ).shape[0] == len(features):
            return _normalize(imp)
    if hasattr(model, "_predictors"):
        imp = np.zeros(len(features), dtype=float)
        for class_trees in model._predictors:
            for tree in class_trees:
                for n in tree.nodes:
                    if not n["is_leaf"] and 0 <= (i := int(n["feature_idx"])) < len(
                        features
                    ):
                        imp[i] += float(n["gain"])
        return _normalize(imp)
    return None


def _permutation_importance(
    model: Any, X: np.ndarray, y: np.ndarray, random_state: int, n_repeats: int
) -> np.ndarray:
    return _normalize(
        permutation_importance(
            model,
            X,
            y,
            n_repeats=max(1, int(n_repeats)),
            random_state=random_state,
            scoring="neg_mean_squared_error",
        ).importances_mean
    )


def select_features(
    model: Any,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    target_col: str = "y",
    method: str = "auto",
    top_k: int = 20,
    cumulative: float = 0.95,
    min_importance: float = 0.0,
    max_samples: int = 20_000,
    n_repeats: int = 2,
    random_state: int = 42,
    out_path: str | Path | None = None,
) -> tuple[list[str], dict]:
    method, rng, source = (
        method.lower().strip(),
        np.random.default_rng(random_state),
        "",
    )
    df = valid_df if not valid_df.empty else train_df
    if 0 < max_samples < len(df):
        df = df.iloc[np.sort(rng.choice(len(df), size=int(max_samples), replace=False))]
    X, y = df[features].to_numpy(dtype=float), df[target_col].to_numpy(dtype=float)

    if method in {"auto", "importance", "gain", "split"}:
        imp, source = _tree_importance(model, features), "native_tree_importance"
        if imp is None:
            imp, source = (
                _permutation_importance(model, X, y, random_state, n_repeats),
                "permutation_importance_fallback",
            )
    elif method == "permutation":
        imp, source = (
            _permutation_importance(model, X, y, random_state, n_repeats),
            "permutation_importance",
        )
    else:
        raise ValueError(
            "feature selection method must be auto, importance, permutation, or none"
        )

    imp = _normalize(imp)
    ranked = [(features[i], float(imp[i])) for i in np.argsort(-imp)]
    selected, cum = [], 0.0
    for n, v in ranked:
        if v < float(min_importance) and selected:
            continue
        selected.append(n)
        cum += v
        if (top_k > 0 and len(selected) >= int(top_k)) or (
            0 < cumulative < 1 and cum >= float(cumulative)
        ):
            break
    selected = selected or [
        features[i]
        for i in np.argsort(-imp)[: max(1, min(int(top_k or 1), len(features)))]
    ]

    report = {
        "method_requested": method,
        "importance_source": source,
        "input_feature_count": len(features),
        "selected_feature_count": len(selected),
        "dropped_feature_count": len(features) - len(selected),
        "top_k": int(top_k),
        "cumulative_importance_target": float(cumulative),
        "min_importance": float(min_importance),
        "max_samples": int(max_samples),
        "n_repeats": int(n_repeats),
        "selected_features": selected,
        "dropped_features": [f for f in features if f not in selected],
        "ranked_importance": [{"feature": n, "importance": v} for n, v in ranked],
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return selected, report
