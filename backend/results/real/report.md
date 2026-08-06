## environment

- python 3.14.0 | Windows-11-10.0.26200-SP0
- CPUs: 20 | GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB
- xgboost 3.3.0 | lightgbm 4.7.0
- data: real | 1,613,233 rows | 103 sensors
- short-history sensors: [184, 185, 187, 188]

## T1_split

train: < 2025-11-01 | val: 2025-11-01..2026-02-28 | test: >= 2026-03-01 (most recent block)

## T2_point

| Horizon | Model | MAE | RMSE | MAPE % | WMAPE % |
| --- | --- | --- | --- | --- | --- |
| 1h | XGB | 59.14 | 136.12 | 46.89 | 14.85 |
| 1h | LGB | 60.87 | 137.29 | 50.32 | 15.28 |
| 6h | XGB | 79.27 | 171.46 | 94.39 | 19.90 |
| 6h | LGB | 80.48 | 170.81 | 96.38 | 20.20 |
| 24h | XGB | 87.01 | 185.90 | 121.73 | 21.83 |
| 24h | LGB | 87.31 | 185.01 | 127.98 | 21.91 |

## T3_intervals

| Horizon | Model | Pinball@0.1 | Pinball@0.5 | Pinball@0.9 | 80% interval pinball | Coverage % | Mean width |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1h | XGB | 13.05 | 28.66 | 16.68 | 14.87 | 74.1 | 161.99 |
| 1h | LGB | 13.35 | 29.39 | 16.57 | 14.96 | 74.9 | 168.35 |
| 6h | XGB | 16.48 | 36.55 | 21.97 | 19.23 | 72.4 | 205.30 |
| 6h | LGB | 16.80 | 37.39 | 21.67 | 19.24 | 72.7 | 207.63 |
| 24h | XGB | 17.73 | 39.82 | 23.97 | 20.85 | 74.1 | 227.81 |
| 24h | LGB | 18.05 | 40.47 | 23.81 | 20.93 | 74.2 | 227.40 |

## T4_traintime

| Training device | XGBoost total (s) | XGBoost mean/model (s) | LightGBM total (s) | LightGBM mean/model (s) |
| --- | --- | --- | --- | --- |
| CPU | 1416 | 118.0 | 243 | 20.3 |
| GPU | 238 | 19.8 | N/A (no GPU build) | - |
| GPU speedup (XGB) | 6.0 | - | - | - |

## T5_latency

| Horizon | Model | Batch us/row | Single ms (mean) | Single ms (p95) | File size (MB) | Serving RSS (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1h | XGB | 0.0032 | 3.799 | 4.773 | 22.021 | 4.0 |
| 1h | LGB | 0.0179 | 1.811 | 2.369 | 5.676 | 0.0 |
| 6h | XGB | 0.0017 | 3.122 | 3.962 | 10.656 | 2.0 |
| 6h | LGB | 0.0047 | 1.080 | 1.188 | 4.174 | 0.0 |
| 24h | XGB | 0.0012 | 3.075 | 4.192 | 7.356 | 1.9 |
| 24h | LGB | 0.0041 | 1.211 | 1.553 | 3.791 | 0.0 |

## T6_groups

| Horizon | Model | Group | n rows | MAE | RMSE | MAPE % | Coverage % | Width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1h | XGB | short | 13622 | 71.55 | 116.97 | 83.63 | 69.3 | 180.62 |
| 1h | XGB | long | 330884 | 58.63 | 136.85 | 45.37 | 74.3 | 161.22 |
| 1h | LGB | short | 13622 | 73.14 | 118.52 | 91.19 | 63.7 | 172.46 |
| 1h | LGB | long | 330884 | 60.37 | 138.01 | 48.64 | 75.4 | 168.18 |
| 6h | XGB | short | 13622 | 98.07 | 150.77 | 195.37 | 67.1 | 223.57 |
| 6h | XGB | long | 330884 | 78.50 | 172.26 | 90.24 | 72.6 | 204.54 |
| 6h | LGB | short | 13622 | 101.81 | 154.06 | 194.26 | 57.7 | 215.81 |
| 6h | LGB | long | 330884 | 79.60 | 171.46 | 92.35 | 73.3 | 207.29 |
| 24h | XGB | short | 13622 | 105.59 | 166.86 | 244.98 | 65.8 | 246.83 |
| 24h | XGB | long | 330884 | 86.25 | 186.64 | 116.66 | 74.5 | 227.03 |
| 24h | LGB | short | 13622 | 113.33 | 178.93 | 349.66 | 60.6 | 221.99 |
| 24h | LGB | long | 330884 | 86.24 | 185.26 | 118.86 | 74.8 | 227.63 |

## T7_importance

| # | Feature | XGBoost gain % | LightGBM gain % |
| --- | --- | --- | --- |
| 1 | lag_24 | 42.89 | 50.81 |
| 2 | lag_168 | 23.90 | 17.09 |
| 3 | lag_1 | 11.35 | 11.63 |
| 4 | hour | 6.38 | 8.08 |
| 5 | roll_24_mean | 5.62 | 5.18 |
| 6 | location_code | 3.63 | 4.81 |
| 7 | is_weekend | 2.76 | 0.11 |
| 8 | dow | 1.72 | 1.25 |
| 9 | roll_168_mean | 0.73 | 0.37 |
| 10 | month | 0.56 | 0.33 |
| 11 | roll_672_mean | 0.48 | 0.34 |
| 12 | is_public_holiday | 0.00 | 0.00 |

Spearman rank corr (gain, all features): 0.900  | Top-10 feature overlap (Jaccard): 0.90
XGBoost top10: ['lag_24', 'lag_168', 'lag_1', 'hour', 'roll_24_mean', 'location_code', 'is_weekend', 'dow', 'roll_168_mean', 'month']
LightGBM top10: ['lag_24', 'lag_168', 'lag_1', 'hour', 'roll_24_mean', 'location_code', 'dow', 'roll_168_mean', 'roll_672_mean', 'month']

## T8_head2head

| Metric | XGBoost | LightGBM | Winner |
| --- | --- | --- | --- |
| MAE @ 1h | 59.14 | 60.87 | XGB |
| RMSE @ 1h | 136.12 | 137.29 | XGB |
| Coverage @ 1h (%) | 74.1 | 74.9 | LGB |
| Interval pinball @ 1h | 14.87 | 14.96 | XGB |
| MAE @ 6h | 79.27 | 80.48 | XGB |
| RMSE @ 6h | 171.46 | 170.81 | LGB |
| Coverage @ 6h (%) | 72.4 | 72.7 | LGB |
| Interval pinball @ 6h | 19.23 | 19.24 | XGB |
| MAE @ 24h | 87.01 | 87.31 | XGB |
| RMSE @ 24h | 185.90 | 185.01 | LGB |
| Coverage @ 24h (%) | 74.1 | 74.2 | LGB |
| Interval pinball @ 24h | 20.85 | 20.93 | XGB |
| CPU training, all 12 models (s) | 1416 | 243 | LGB |
| GPU training, all 12 models (s) | 238 | N/A (pip wheel has no GPU) | - |
| Batch inference us/row | 0.0012 | 0.0041 | - |
| Single-row inference ms (mean) | 4.120 | 1.446 | - |
| Model file size (point @1h, MB) | 22.938 | 6.068 | - |
| Serving RSS delta (MB) | 2.6 | 0.0 | - |
| GPU support | CUDA (pip) | N/A (needs source build) | - |