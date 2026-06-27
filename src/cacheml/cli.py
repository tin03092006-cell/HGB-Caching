from __future__ import annotations
from .common import *


from .config import CacheMLConfig
from .data import load_trace
from .models import split_train_valid_test, train_model, save_model, default_params
from .preprocess import prepare, FEATURES
from .simulator import simulate_lru, simulate_lfu, simulate_belady, simulate_ml_replacement
from .system_power import set_full_cpu_threads, build_hardware_plan, save_plan
from .tuning import tune_optuna
from .feature_selection import select_features
from .imbalance import make_training_matrix, save_imbalance_report
from .rawdata import convert_twitter


def _cfg_from_args(args) -> CacheMLConfig:
    cfg = CacheMLConfig()
    mapping = {
        "workers": ("workers", int),
        "latency_us": ("latency_budget_us", float),
        "ram_fraction": ("ram_fraction", float),
        "reserve_ram_mb": ("reserve_ram_mb", float),
        "candidate_n": ("candidate_n", int),
        "eviction_batch": ("eviction_batch", int),
        "max_rows": ("max_rows", int),
        "imbalance_strategy": ("imbalance_strategy", str),
        "downsample_max_per_object": ("downsample_max_per_object", int),
        "sample_weight_max": ("sample_weight_max", float),
        "hot_weight_strength": ("hot_weight_strength", float),
        "rare_weight_strength": ("rare_weight_strength", float),
        "size_weight_strength": ("size_weight_strength", float),
    }
    for arg_attr, (cfg_attr, type_func) in mapping.items():
        val = getattr(args, arg_attr, None)
        if val is not None:
            setattr(cfg, cfg_attr, type_func(val))
    return cfg

def cmd_convert(args):
    convert_twitter(args.raw, args.out, max_rows=args.max_rows)

def cmd_prepare(args):
    cfg = _cfg_from_args(args)
    workers = set_full_cpu_threads(cfg.n_workers)
    df = load_trace(args.trace, max_rows=cfg.max_rows)
    print(f"rows={len(df)} workers={workers}")
    out, meta = prepare(df, cfg)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(p)
    meta_path = p.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {p}")


def _do_feature_selection(args, cfg, model, train_df, valid_df, all_features):
    if args.feature_select == "none":
        return model, default_params(cfg), all_features
    selected_features, fs_report = select_features(
        model=model,
        train_df=train_df,
        valid_df=valid_df,
        features=all_features,
        method=args.feature_select,
        top_k=args.feature_top_k,
        cumulative=args.feature_cumulative,
        min_importance=args.feature_min_importance,
        max_samples=args.feature_select_samples,
        n_repeats=args.feature_permutation_repeats,
        random_state=cfg.random_state,
        out_path=Path(args.results_dir) / "feature_selection.json",
    )
    print(f"feature_selection={fs_report['importance_source']} selected={len(selected_features)}/{len(all_features)}")
    print("selected_features=" + ",".join(selected_features))
    model, params = train_model(train_df, cfg, default_params(cfg), features=selected_features)
    return model, params, selected_features


def _do_optuna_tuning(args, cfg, df, train_df, valid_df, W, alpha, candidate_n, eviction_batch, Md_bound, selected_features, meta):
    best_params, best_val = tune_optuna(
        train_df=train_df,
        valid_df=valid_df,
        cfg=cfg,
        trials=args.trials,
        cache_size=args.cache_size,
        W=W,
        alpha=alpha,
        candidate_n=candidate_n,
        eviction_batch=eviction_batch,
        Md_bound=int(Md_bound),
        features=selected_features,
    )
    print(f"optuna_best_byte_hit_rate={best_val:.6f}")
    print(json.dumps(best_params, indent=2))
    model, params = train_model(pd.concat([train_df, valid_df]), cfg, best_params, features=selected_features)
    X_probe = train_df[selected_features].to_numpy(dtype=float)[: min(4096, len(train_df))]
    plan = build_hardware_plan(cfg, df, model=model, X_probe=X_probe, W_acf=meta.get("W_acf"), record_bytes=meta.get("record_bytes"))
    return model, params, plan


def _run_simulations(args, cfg, test_df, model, W, alpha, candidate_n, eviction_batch, selected_features):
    rows = [
        simulate_belady(test_df, args.cache_size).to_dict(),
        simulate_lru(test_df, args.cache_size).to_dict(),
        simulate_lfu(test_df, args.cache_size).to_dict(),
        simulate_ml_replacement(
            test_df,
            cache_size=args.cache_size,
            model=model,
            W=W,
            alpha=alpha,
            candidate_n=candidate_n,
            eviction_batch=eviction_batch,
            random_state=cfg.random_state,
            features=selected_features,
        ).to_dict(),
    ]
    res = pd.DataFrame(rows).sort_values("hit_rate", ascending=False)
    out_csv = Path(args.results_dir) / "benchmark.csv"
    res.to_csv(out_csv, index=False)
    print("\nBenchmark:")
    print(res.to_string(index=False))
    print(f"\nwrote {out_csv}")
    return res


def cmd_bench(args):
    cfg = _cfg_from_args(args)
    set_full_cpu_threads(cfg.n_workers)
    df = pd.read_pickle(args.data)
    if cfg.max_rows and cfg.max_rows > 0:
        df = df.iloc[: cfg.max_rows].copy()
    print(f"Loaded {len(df)} rows. Imbalance strategy: {cfg.imbalance_strategy}")
    train_df, valid_df, test_df = split_train_valid_test(df)
    print(f"split train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    _, _, _, _, imbalance_report = make_training_matrix(train_df, list(FEATURES), cfg)
    save_imbalance_report(imbalance_report, Path(args.results_dir) / "imbalance_report.json")
    print("imbalance_strategy=" + str(cfg.imbalance_strategy) + " downsample_max_per_object=" + str(cfg.downsample_max_per_object))

    meta_path = Path(args.data).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    alpha = float(meta.get("alpha", 0.1))

    all_features = list(FEATURES)
    model, params = train_model(train_df, cfg, default_params(cfg), features=all_features)

    # Feature selection extraction
    model, params, selected_features = _do_feature_selection(args, cfg, model, train_df, valid_df, all_features)

    X_probe = train_df[selected_features].to_numpy(dtype=float)[: min(4096, len(train_df))]
    plan = build_hardware_plan(cfg, df, model=model, X_probe=X_probe, W_acf=meta.get("W_acf"), record_bytes=meta.get("record_bytes"))
    W = int(meta.get("W", plan["W_final"]))
    candidate_n = int(cfg.candidate_n or plan["candidate_n"])
    eviction_batch = int(cfg.eviction_batch or plan["eviction_batch"])

    # Optuna tuning extraction
    if args.trials and args.trials > 0:
        model, params, plan = _do_optuna_tuning(
            args, cfg, df, train_df, valid_df, W, alpha, candidate_n, eviction_batch,
            plan["model_product_Md_bound"], selected_features, meta
        )
        candidate_n = int(cfg.candidate_n or plan["candidate_n"])
        eviction_batch = int(cfg.eviction_batch or plan["eviction_batch"])

    plan.update({
        "feature_count_all": len(all_features),
        "feature_count_selected": len(selected_features),
        "selected_features": selected_features,
        "imbalance_strategy": cfg.imbalance_strategy,
        "downsample_max_per_object": cfg.downsample_max_per_object,
        "sample_weight_max": cfg.sample_weight_max,
        "hot_weight_strength": cfg.hot_weight_strength,
        "rare_weight_strength": cfg.rare_weight_strength,
        "size_weight_strength": cfg.size_weight_strength,
    })
    
    if args.model_out:
        model_path = Path(args.model_out)
    else:
        model_path = Path("models") / Path(args.results_dir).name / "cacheml.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, params, str(model_path), features=selected_features)
    save_plan(plan, Path(args.results_dir) / "hardware_plan.json")
    print(f"wrote {Path(args.results_dir) / 'hardware_plan.json'}")

    # Simulation extraction
    _run_simulations(args, cfg, test_df, model, W, alpha, candidate_n, eviction_batch, selected_features)


def _add_common_args(parser):
    parser.add_argument("--max-rows", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    parser.add_argument("--ram-fraction", type=float, default=0.85)
    parser.add_argument("--reserve-ram-mb", type=float, default=1024.0)


def build_parser():
    p = argparse.ArgumentParser("cacheml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare")
    s.add_argument("--trace", required=True)
    s.add_argument("--out", required=True)
    _add_common_args(s)
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("convert")
    s.add_argument("--raw", required=True, help="Path to raw twitter text trace (e.g. cluster045)")
    s.add_argument("--out", required=True, help="Path to output CSV")
    s.add_argument("--max-rows", type=int, default=0, help="0 = full dataset")
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("bench")
    s.add_argument("--data", required=True)
    s.add_argument("--cache-size", type=int, required=True)
    s.add_argument("--feature-select", default="auto", choices=["none", "auto", "importance", "permutation"], help="prune weak features before Optuna/final training")
    s.add_argument("--feature-top-k", type=int, default=20, help="max selected features; 0 = no top-k limit")
    s.add_argument("--feature-cumulative", type=float, default=0.95, help="keep features until cumulative importance reaches this value; 0/1 disables")
    s.add_argument("--feature-min-importance", type=float, default=0.0)
    s.add_argument("--feature-select-samples", type=int, default=20000)
    s.add_argument("--feature-permutation-repeats", type=int, default=2)
    s.add_argument("--imbalance-strategy", default="hybrid", choices=["none", "hot", "rare", "balanced", "hybrid"], help="sample weighting for Zipfian object imbalance")
    s.add_argument("--downsample-max-per-object", type=int, default=0, help="training-only cap per object; 0 disables downsampling")
    s.add_argument("--sample-weight-max", type=float, default=20.0, help="cap sample weights to avoid unstable fitting")
    s.add_argument("--hot-weight-strength", type=float, default=0.50)
    s.add_argument("--rare-weight-strength", type=float, default=0.75)
    s.add_argument("--size-weight-strength", type=float, default=0.25)
    s.add_argument("--trials", type=int, default=0)
    s.add_argument("--latency-us", type=float, default=1000.0)
    s.add_argument("--candidate-n", type=int, default=0, help="0 = auto from hardware")
    s.add_argument("--eviction-batch", type=int, default=0, help="0 = auto from hardware")
    s.add_argument("--results-dir", default="results")
    s.add_argument("--model-out", default="", help="If not set, mirrors results-dir inside models/")
    _add_common_args(s)
    s.set_defaults(func=cmd_bench)
    return p
