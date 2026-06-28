# HGB-Caching

**ML-Augmented Online Cache Replacement using HistGradient Boosting**  
Course project for **Introduction to Artificial Intelligence (IT3160)**, Hanoi University of Science and Technology.

> HGB-Caching studies whether a lightweight machine-learning model can improve online cache replacement decisions by predicting the next access distance of cached objects.

---

## 1. Motivation

Caches appear in CPU memory hierarchy, operating systems, databases, content delivery networks, API gateways and edge computing systems. Because cache capacity is limited, a system must decide which object to evict when a cache miss occurs and the cache is already full.

Classic policies such as **LRU** and **LFU** rely only on fixed hand-written heuristics. The optimal offline policy, **Belady**, evicts the object whose next use is farthest in the future, but Belady requires the entire future request trace and therefore cannot be used in a real online system.

This project follows the learning-augmented idea: use machine learning to estimate a future-related quantity, while the simulator still makes decisions under an online setting.

---

## 2. Problem setting

At time step `t`, the request is `r_t` and the current cache state is `C_t`.

- If `r_t` is already in `C_t`, the request is a **cache hit**.
- If `r_t` is not in `C_t`, the request is a **cache miss**.
- If a miss occurs and `|C_t| = K`, the policy must choose a victim object `v_t ∈ C_t` to evict.

The simulator currently uses **object-capacity caching**:

```text
|C_t| <= K
```

where `K` is the maximum number of cached objects. The current system does not enforce a byte-capacity constraint. Byte-level performance is still reported through **Byte Hit Rate**, but cache capacity itself is counted by number of objects.

Admission is simple and fixed: when a miss occurs and `K > 0`, the requested object is always admitted into the cache. Therefore, this project focuses on **eviction**, not admission control.

---

## 3. Main idea

The machine-learning policy tries to imitate the key signal used by Belady: **next access distance**.

For an object requested at time `t`, its natural target is

```text
d_t = min { j - t | j > t and o_j = o_t }
```

If the object does not appear again in the remaining trace, the implementation uses

```text
d_t = n + 1 - t
```

The model is trained as a regression model. During eviction, it predicts the future reuse distance of candidate cached objects and evicts the object with the largest predicted distance.

```text
v_t = argmax_{o in S_t} d_hat_t(o)
```

where `S_t` is the candidate set sampled from the current cache.

---

## 4. ML policy

When a miss occurs and the cache is full, the ML policy does the following:

1. Build a candidate set `S_t ⊆ C_t`.
2. Compute online features for every object in `S_t` using only history before time `t`.
3. Predict next access distance with `HistGradientBoostingRegressor`.
4. Evict the candidate object with the largest predicted distance.

If `candidate_n < |C_t|`, the simulator randomly samples `candidate_n` objects from the cache. If `candidate_n >= |C_t|`, the candidate set is the whole cache.

This keeps inference cost bounded while preserving the online nature of the decision.

---

## 5. Model

The core model is **HistGradientBoostingRegressor** from scikit-learn.

The ensemble has the form

```text
y_hat(x) = sum_{m=1}^{M} nu * f_m(x)
```

where each tree `f_m` is trained sequentially to reduce the residual error of the previous ensemble.

Important implementation details:

- `n_estimators` in the project config is mapped to sklearn's `max_iter`.
- `num_leaves` is mapped to sklearn's `max_leaf_nodes`.
- `early_stopping=True` is enabled.
- Optuna can tune hyperparameters using validation **Byte Hit Rate** as the objective.

---

## 6. Target transform

The natural target `d_t` can be extremely skewed because real traces often contain hot objects, rare objects and long-tail access patterns.

The preprocessing stage applies a Box-Cox transform to the natural next access distance target:

```python
y_bc, lambda_ = scipy.stats.boxcox(y_nat)
```

If the target is almost constant, the implementation falls back to

```python
y = log1p(y_nat)
```

Because Box-Cox is monotonic for positive values, inverse-transform is not required during eviction if the policy only needs to rank candidates by predicted distance.

---

## 7. Online features

Features are computed from the state **before** the current request updates the history. This avoids future leakage.

The project currently uses 29 online features:

```text
size, log_size, type_code, op_code,
ttl, log_ttl, recency, log_recency,
count_short, count_mid, count_long, ewma_freq,
log_freq_seen, freq_density, last_gap, prev_gap,
gap_ewma, gap_mean, gap_std, gap_cv,
burst_ratio, acceleration, unique_since_last, log_unique_since_last,
byte_freq, byte_ewma, byte_count_w, age_seen,
log_age_seen
```

Examples:

| Feature | Meaning |
|---|---|
| `recency` | Number of steps since the object was last requested. |
| `ewma_freq` | Exponentially weighted frequency from past requests. |
| `freq_density` | Number of observations divided by observed age. |
| `unique_since_last` | Number of distinct objects seen since the object's previous request. |
| `gap_ewma` | EWMA-smoothed inter-arrival gap. |
| `gap_cv` | Coefficient of variation of previous gaps. |
| `size` | Request/object size from the trace. |

---

## 8. Automatic history window and EWMA decay

The preprocessing stage estimates two workload-dependent quantities:

### History window `W`

```text
W = min(W_ACF, W_RAM)
```

- `W_ACF`: the first autocorrelation lag that falls below a configured threshold.
- `W_RAM`: the largest number of records allowed by the available RAM budget.

### EWMA decay `alpha`

`alpha` is estimated from the mean gap of the most frequent hot objects:

```text
alpha = 1 - exp(-1 / max(mean_gap_hot, 1))
```

This lets the feature extractor adapt to the locality dynamics of the workload instead of using a fixed hand-picked decay constant.

---

## 9. Imbalance handling

Real cache traces are highly imbalanced: a small number of hot objects can dominate the request count while many objects appear only once or a few times.

The project supports the following sample-weight strategies:

- `none`
- `hot`
- `rare`
- `balanced`
- `hybrid`

The `hybrid` strategy combines frequency-aware, rare-object-aware and size-aware weighting. Weights are clipped by `sample_weight_max` and normalized by their mean.

This `hybrid` mode is a **training sample-weighting strategy**, not a separate ML-plus-heuristic eviction policy.

---

## 10. Feature selection

The project supports feature selection before Optuna tuning and final training.

Available modes:

- `none`
- `auto`
- `importance`
- `permutation`

The output report contains:

- selected feature count,
- dropped feature count,
- selected feature list,
- ranked feature importance.

---

## 11. Hardware-aware candidate planning

The simulator estimates a hardware plan using CPU, RAM and prediction latency.

Important outputs include:

- available RAM budget,
- estimated request rate,
- prediction latency per object,
- `candidate_n`,
- `eviction_batch`,
- model-depth/tree bound for latency-aware tuning.

The goal is not to prove real-time production readiness, but to make the simulation aware of inference cost.

---

## 12. Repository structure

```text
src/cacheml/
├── cli.py                 # Command-line interface
├── common.py              # Shared imports/utilities
├── config.py              # Central configuration dataclass
├── data.py                # Load standardized CSV trace
├── feature_selection.py   # Native tree/permutation feature selection
├── feature_utils.py       # Fenwick tree and feature helpers
├── imbalance.py           # Sample weighting and downsampling
├── main.py                # CLI entry point
├── models.py              # Split data and train HGB model
├── preprocess.py          # Online features, NAD label, Box-Cox target
├── rawdata.py             # Convert Twitter raw trace to project CSV
├── simulator.py           # Belady, LRU, Global-LFU and ML simulation
├── system_power.py        # CPU/RAM/latency planning
└── tuning.py              # Optuna tuning by validation Byte Hit Rate
```

Typical generated directories:

```text
data/       # raw and processed traces
models/     # trained joblib models
results/    # benchmark.csv, best_params.json, hardware_plan.json, reports
```

---

## 13. Installation

Python version:

```text
Python >= 3.10
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Main dependencies:

- numpy
- pandas
- scipy
- scikit-learn
- joblib
- psutil
- optuna
- tqdm
- zstandard

---

## 14. Data format

The standardized CSV format is:

```text
obj,size,type,timestamp,op,ttl
```

For Twitter Block I/O traces, the converter expects raw rows in the following format:

```text
timestamp,key,key_size,value_size,client_id,op,ttl
```

The converter maps it to:

| Output column | Source |
|---|---|
| `obj` | `key` |
| `size` | `key_size + value_size` |
| `type` | `client_id` |
| `timestamp` | `timestamp` |
| `op` | operation field |
| `ttl` | TTL field |

---

## 15. Running the pipeline

### Step 1: Convert raw trace

```bash
python -m cacheml.main convert \
  --raw data/raw/cluster045 \
  --out data/processed/twitter45.csv
```

### Step 2: Prepare features and target

```bash
python -m cacheml.main prepare \
  --trace data/processed/twitter45.csv \
  --out data/processed/twitter45.pkl \
  --workers 0
```

This creates:

```text
data/processed/twitter45.pkl
data/processed/twitter45.meta.json
```

### Step 3: Run benchmark

Example for `K = 100`, hybrid sample weighting and 15 Optuna trials:

```bash
python -m cacheml.main bench \
  --data data/processed/twitter45.pkl \
  --cache-size 100 \
  --imbalance-strategy hybrid \
  --trials 15 \
  --feature-select auto \
  --results-dir results/twitter45_c100_hybrid_optuna15
```

Repeat for other cache sizes:

```bash
python -m cacheml.main bench --data data/processed/twitter45.pkl --cache-size 250 --imbalance-strategy hybrid --trials 15 --feature-select auto --results-dir results/twitter45_c250_hybrid_optuna15
python -m cacheml.main bench --data data/processed/twitter45.pkl --cache-size 500 --imbalance-strategy hybrid --trials 15 --feature-select auto --results-dir results/twitter45_c500_hybrid_optuna15
```

To compare with no sample weighting:

```bash
python -m cacheml.main bench --data data/processed/twitter45.pkl --cache-size 100 --imbalance-strategy none --trials 15 --feature-select auto --results-dir results/twitter45_c100_none_optuna15
python -m cacheml.main bench --data data/processed/twitter45.pkl --cache-size 250 --imbalance-strategy none --trials 15 --feature-select auto --results-dir results/twitter45_c250_none_optuna15
python -m cacheml.main bench --data data/processed/twitter45.pkl --cache-size 500 --imbalance-strategy none --trials 15 --feature-select auto --results-dir results/twitter45_c500_none_optuna15
```

---

## 16. Outputs

Each benchmark run writes files under `results/<run_name>/`.

Important files:

| File | Meaning |
|---|---|
| `benchmark.csv` | Final comparison among Belady, LRU, Global-LFU and ML. |
| `best_params.json` | Best Optuna trial and hyperparameters. |
| `hardware_plan.json` | CPU/RAM/latency-aware candidate plan. |
| `feature_selection.json` | Selected features and importance ranking. |
| `imbalance_report.json` | Object-frequency imbalance and sample-weight report. |

The trained model is saved under `models/<run_name>/cacheml.joblib` unless `--model-out` is specified.

---

## 17. Metrics

### Hit Rate

```text
HR = #Hits / (#Hits + #Misses)
```

Hit Rate counts requests.

### Byte Hit Rate

```text
BHR = Bytes Hit / Total Bytes Requested
```

Byte Hit Rate counts the amount of data served from cache.

### Average ML inference latency

The current simulator reports prediction latency only for the ML policy. For Belady, LRU and Global-LFU, `0.00` means **not measured**, not zero-cost execution.

---

## 18. Reported benchmark snapshot

The following values are the reported results for the processed Twitter Cluster045 workload with 15,000 test requests per benchmark.

| Cache size | Policy | Hit Rate | Byte Hit Rate | Avg ML latency | AI calls |
|---:|---|---:|---:|---:|---:|
| 100 | Belady | 8.31% | 16.15% | not measured | 0 |
| 100 | ML-HGB hybrid | 5.91% | 11.64% | 3270.70 us | 93,874 |
| 100 | Global-LFU | 5.65% | 10.91% | not measured | 0 |
| 100 | LRU | 2.27% | 4.18% | not measured | 0 |
| 250 | Belady | 10.64% | 20.41% | not measured | 0 |
| 250 | ML-HGB hybrid | 7.54% | 15.18% | 2886.21 us | 92,443 |
| 250 | Global-LFU | 7.27% | 14.39% | not measured | 0 |
| 250 | LRU | 3.34% | 6.25% | not measured | 0 |
| 500 | Belady | 12.48% | 23.54% | not measured | 0 |
| 500 | ML-HGB hybrid | 8.69% | 17.46% | 5194.78 us | 91,325 |
| 500 | Global-LFU | 8.56% | 16.69% | not measured | 0 |
| 500 | LRU | 4.05% | 7.67% | not measured | 0 |

Observed from this snapshot:

- ML-HGB beats LRU clearly on all reported cache sizes.
- ML-HGB beats Global-LFU slightly on Byte Hit Rate.
- Belady remains a much stronger offline oracle upper bound.
- The current Python implementation has high ML inference latency, so further engineering is required before real-time deployment.

Approximate Byte-Hit-Rate closed gap from Global-LFU to Belady:

| Cache size | Closed gap |
|---:|---:|
| 100 | 13.93% |
| 250 | 13.12% |
| 500 | 11.24% |

---

## 19. Fairness protocol

The benchmark uses the following protocol:

- All policies are evaluated on the same sequential test segment.
- Data is split by time as `70% train / 15% validation / 15% test`.
- No shuffle is used, to avoid future leakage.
- Belady is included only as an offline oracle reference.
- LRU and Global-LFU are online non-ML baselines.
- ML-HGB is online at decision time: features use only historical state.
- Admission is fixed: every missed object is admitted if `K > 0`.
- ML candidate sampling is randomized with a fixed random seed from config.

---

## 20. Limitations

This project is a course-scale experimental system, not a production cache.

Current limitations:

1. The simulator uses object capacity, not byte capacity.
2. The strongest reported benchmark is mainly on the processed Twitter Cluster045 workload.
3. Candidate sampling introduces randomness; repeated multi-seed evaluation would be better.
4. ML inference latency is still high in the Python implementation.
5. The current model is trained offline and does not perform continuous online learning.
6. More regression/ranking diagnostics should be reported, such as MAE, RMSE and Spearman correlation for predicted next access distance.
7. More real-world traces are needed to confirm generalization.

---

## 21. Future work

Possible extensions:

- Add byte-capacity simulation.
- Add multi-seed benchmark with confidence intervals.
- Add ranking metrics for eviction quality.
- Add more traces from different workloads.
- Optimize inference with batching, model compression or a lower-latency implementation.
- Study admission policies instead of always admitting every missed object.
- Add online adaptation for workload drift.

---

## 22. References

- L. A. Belady, *A study of replacement algorithms for a virtual-storage computer*, IBM Systems Journal, 1966.
- F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 2011.
- T. Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD, 2019.
- Twitter Block I/O Traces, Storage Performance Council / SNIA Data Storage Traces Repository.
