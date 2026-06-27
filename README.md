# CacheML Replacement Fullpower — Feature v2 + Pruning + Zipf Imbalance Handling

Luật benchmark chính giữ nguyên:

- Không có admission control.
- Không được từ chối nạp object.
- `rejected = 0`.
- Miss + cache đầy → chọn victim → evict → bắt buộc insert object hiện tại.

## Điểm mới

Bản này bổ sung xử lý mất cân bằng Zipfian trong training:

- `--imbalance-strategy hybrid`: vừa giữ tín hiệu hot objects, vừa bảo vệ rare objects, vừa có size-aware weight.
- `--imbalance-strategy hot`: giống bản cũ, nhấn mạnh object xuất hiện nhiều.
- `--imbalance-strategy rare`: nhấn mạnh object hiếm.
- `--imbalance-strategy balanced`: cân bằng ảnh hưởng giữa các object bằng `1/sqrt(freq)`.
- `--downsample-max-per-object K`: down-sampling training-only, giới hạn mỗi object đóng góp tối đa K dòng.
- Luôn sinh `imbalance_report.json` để kiểm tra phân phối trước/sau downsample và weight min/mean/max.

Feature pruning vẫn có:

- `--feature-select permutation`
- `--feature-top-k 10/15/20`

## Chạy lại project

```powershell
python -m pip install -e .
```

## Prepare lại nếu cần

```powershell
python -m cacheml.main prepare --trace data/twitter45.csv --out data/twitter45_v2.prepared.pkl --workers 0 --ram-fraction 0.85 --reserve-ram-mb 1024
```

## Benchmark khuyến nghị: hybrid weights + downsampling hot objects

```powershell
python -m cacheml.main bench --data data/twitter45_v2.prepared.pkl --cache-size 1000 --model hgb --trials 10 --workers 0 --latency-us 2000 --candidate-n 90 --eviction-batch 16 --feature-select permutation --feature-top-k 15 --feature-select-samples 20000 --feature-permutation-repeats 2 --imbalance-strategy hybrid --downsample-max-per-object 200 --sample-weight-max 20 --results-dir results\tw_v2_pruned_imbalance_optuna10
```

Benchmark thật:

```powershell
python -m cacheml.main bench --data data/twitter45_v2.prepared.pkl --cache-size 1000 --model hgb --trials 30 --workers 0 --latency-us 2000 --candidate-n 90 --eviction-batch 16 --feature-select permutation --feature-top-k 15 --feature-select-samples 20000 --feature-permutation-repeats 2 --imbalance-strategy hybrid --downsample-max-per-object 200 --sample-weight-max 20 --results-dir results\tw_v2_pruned_imbalance_optuna30
```

## So sánh ablation nên chạy

Không xử lý mất cân bằng:

```powershell
python -m cacheml.main bench --data data/twitter45_v2.prepared.pkl --cache-size 1000 --model hgb --trials 10 --workers 0 --latency-us 2000 --candidate-n 90 --eviction-batch 16 --feature-select permutation --feature-top-k 15 --imbalance-strategy none --downsample-max-per-object 0 --results-dir results\tw_ablate_no_imbalance
```

Chỉ weight, không downsample:

```powershell
python -m cacheml.main bench --data data/twitter45_v2.prepared.pkl --cache-size 1000 --model hgb --trials 10 --workers 0 --latency-us 2000 --candidate-n 90 --eviction-batch 16 --feature-select permutation --feature-top-k 15 --imbalance-strategy hybrid --downsample-max-per-object 0 --results-dir results\tw_ablate_weight_only
```

Weight + downsample:

```powershell
python -m cacheml.main bench --data data/twitter45_v2.prepared.pkl --cache-size 1000 --model hgb --trials 10 --workers 0 --latency-us 2000 --candidate-n 90 --eviction-batch 16 --feature-select permutation --feature-top-k 15 --imbalance-strategy hybrid --downsample-max-per-object 200 --results-dir results\tw_ablate_weight_downsample
```

## Xem report

```powershell
type results\tw_v2_pruned_imbalance_optuna30\benchmark.csv
type results\tw_v2_pruned_imbalance_optuna30\feature_selection.json
type results\tw_v2_pruned_imbalance_optuna30\imbalance_report.json
type results\tw_v2_pruned_imbalance_optuna30\hardware_plan.json
```
