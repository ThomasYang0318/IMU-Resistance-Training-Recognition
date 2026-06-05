# 014 文獻啟發 Rep 切割方法比較

> 2026-05-17 artifact cleanup note：完整 `waveform_all_sets/` 已依文件瘦身政策刪除；保留 summary、核心 CSV 與主圖。若需要全量 set waveform 圖，請用原工具重新產生。

## 目的

第 014 版把前面討論的幾篇 repetition counting / waveform segmentation 論文，轉成可以在目前 IMU 資料上直接比較的 active-only rep boundary 方法。這版不處理休息段，也不做 upstream active detection；輸入假設已知道這段是某個動作的 active set，目標是把 rep boundary 切準。

評估資料：

```text
sessions = 14
active blocks = 239
true sets = 236
true reps = 2720
input axes = ax ay az gx gy gz mx my mz
```

## 方法設計

| 方法 | 文獻啟發 | 實作重點 | 主要弱點 |
|---|---|---|---|
| STAYFIT-BA | StayFit 類 best-axis / peak counting | 每個 active set 自動選最有週期性的 IMU 軸或 magnitude，再用 peak midpoint 切 rep | 單軸容易受配戴方向與 phase wiggle 影響 |
| MAXXYT-MAP | Maxxyt 類 multi-axis adaptive peak aggregation | 多軸 peak count 投票，修正倍頻 over-count，再用 gyro valley 精修 boundary | count 穩，但 boundary 需要額外精修 |
| MFIT-FSTE | M-Fitness 類 short-time frequency-weighted energy | 用 PCA、acc magnitude、gyro magnitude 建能量曲線，再找低能量 valley | 能量峰不一定一個 rep 對一個峰 |
| CARA-DTW-FS | CaRaCount / DTW 類 few-shot template | 用同 subject/exercise 前幾下標註 rep 建 template，再做 shape-aware boundary refinement | 需要少量新人標註；預設用快速 template distance，完整 DTW 需加 `--use-dtw-shape-cost` |
| LIFT-Fusion | 本專案新方法 | 融合 PCA/autocorr 週期、多軸 count consensus、短時能量/gyro valley、few-shot template | 還依賴 active-only/exercise hint，尚未解 upstream set segmentation |

LIFT-Fusion 的核心流程：

```text
9-axis IMU active set
-> PCA principal motion + autocorrelation 估週期
-> 多軸 peak count consensus 估 rep 數
-> short-time energy + gyro valley 建 boundary score
-> 少量標註 template 做個人化 shape prior
-> 輸出 rep boundary
-> 用 IoU@0.50 / 0.75 / 0.90 評估
```

## 主要結果

和既有 010/011/012 放在同一張表比較：

| Method | Count exact | Count +/-1 | Count MAE | Rep F1@0.50 | Rep F1@0.75 | Rep F1@0.90 | Phase F1@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UGV: Universal Gyro Valley | 0.5678 | 0.8347 | 1.7034 | 0.7278 | 0.3949 | 0.1626 | 0.4552 |
| MFBS: Multi-Feature Boundary Score | 0.4110 | 0.7797 | 1.7288 | 0.7382 | 0.4106 | 0.1621 | 0.4736 |
| 9A-DS-MS-TCN-EO | 0.2415 | 0.3771 | 5.7966 | 0.4765 | 0.2890 | 0.1301 | 0.4013 |
| STAYFIT-BA | 0.7119 | 0.9237 | 1.4110 | 0.7343 | 0.4213 | 0.1686 | 0.5069 |
| MAXXYT-MAP | 0.6822 | 0.9364 | 0.6186 | 0.7648 | 0.4380 | 0.2017 | 0.5305 |
| MFIT-FSTE | 0.6271 | 0.8898 | 1.0381 | 0.5983 | 0.2082 | 0.0463 | 0.3344 |
| CARA-DTW-FS | 0.2500 | 0.7203 | 1.2458 | 0.7280 | 0.3968 | 0.1685 | 0.4610 |
| LIFT-Fusion | 0.6737 | 0.9280 | 0.6314 | 0.7855 | 0.4610 | 0.1920 | 0.5319 |

結論：

- LIFT-Fusion 是目前整體 boundary F1 最好的方法：`IoU@0.50 F1 = 0.7855`、`IoU@0.75 F1 = 0.4610`。
- MAXXYT-MAP 的 count-level 最穩：`count +/-1 = 0.9364`、`count MAE = 0.6186 reps`，而且 `IoU@0.90 F1 = 0.2017` 略高於 LIFT。
- LIFT-Fusion 相比 011 MFBS：`IoU@0.50` 從 `0.7382` 到 `0.7855`，`IoU@0.75` 從 `0.4106` 到 `0.4610`，`IoU@0.90` 從 `0.1621` 到 `0.1920`。
- 仍沒有達到「IoU@0.90 F1 90%」目標。現在比較像是 count 已接近可用，但 boundary 精準度還不夠做高可信 TUT / 向心離心細切。

## 動作與人的弱點

LIFT-Fusion 每動作 IoU@0.75 F1：

```text
one_arm_db_row       0.6821
db_biceps_curl       0.5767
db_squat             0.5150
db_triceps_curl      0.4699
db_weighted_crunch   0.4122
db_rdl               0.4024
db_bench_press       0.3185
db_shoulder_press    0.3054
```

LIFT-Fusion 每人 IoU@0.75 F1 顯示 `yoru0511workout`、`yanz0510workout` 較弱，代表現在的錯誤不是單純方法問題，也有明顯 subject-specific waveform 差異。

## 輸出檔案

```text
artifacts_rep_classification/014_literature_inspired_rep_methods/
  014_literature_method_comparison.csv
  014_literature_method_comparison_with_prior.csv
  014_literature_method_comparison_table.png
  014_literature_method_score_bars.png
  014_method_exercise_f1_iou_0.50.png
  014_method_exercise_f1_iou_0.75.png
  014_method_exercise_f1_iou_0.90.png
  014_method_subject_f1_iou_0.50.png
  014_method_subject_f1_iou_0.75.png
  014_method_subject_f1_iou_0.90.png
  waveform_all_sets/*.png
  methods/<method_id>/
```

`waveform_all_sets/` 共輸出 `239` 張圖。每張圖同時放 ground truth 與五種方法，只畫切割線，不塗底色。

## 下一步

若目標是 IoU@0.90 F1 接近 90%，下一步不應只再加特徵，而要處理三件事：

1. boundary candidate recall：先量化 GT boundary ±5 / ±10 / ±20 samples 內有沒有候選點；
2. subject calibration：新人少量標註後估計個人 duration scale、boundary offset、axis weights；
3. phase-aware decoding：rep boundary 和向心/離心 reversal 同時解，而不是先切 rep 再切 phase。
