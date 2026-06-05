# 023 CE Phase-Aware Fatigue 與 RPE 驗證

## 目的

驗證以下猜想是否成立：

> IMU 在疲勞狀態下，應該不只是整段 rep 變不穩，而是 concentric / eccentric phase 會出現特定方向的變化；這些變化是否和 Borg/RPE 有關？

本版使用 ground-truth CE phase 標記，不使用 predicted segmentation，避免切割誤差干擾疲勞特徵判斷。

## 輸入

```text
datasets/workout/*whole_session*.csv
artifacts_rep_classification/018_borg_gt_waveform_relation/018_borg_targets_completed.csv
```

## 輸出

```text
artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/
```

主要檔案：

- `023_phase_aware_rep_feature_dataset.csv`
- `023_phase_aware_set_feature_dataset.csv`
- `023_phase_aware_rep_correlations.csv`
- `023_phase_aware_set_correlations.csv`
- `023_phase_aware_set_correlations_by_exercise.csv`
- `023_phase_aware_hypothesis_bars.png`
- `023_phase_aware_set_top_raw_spearman.png`
- `023_phase_aware_set_top_subject_exercise_centered_spearman.png`

## 資料量

```text
rep rows = 1677
set rows = 143
subjects = haoyu, hsianshun, tsenyu, yanz, yoru, yushuan
```

## 方法

每一下 rep 依照 ground-truth `phase` 拆成：

```text
concentric phase
eccentric phase
```

每個 phase 計算：

- duration；
- PCA range；
- PCA movement rate；
- gyro diff RMS；
- gyro peak；
- acc diff RMS；
- phase waveform similarity to first 2 reps；
- CE ratio。

set-level 再計算：

- mean；
- first2；
- last2；
- slope；
- last2 vs first2 change；
- CV。

本版已修正為「整段 session 先標準化，再切 CE phase」，避免每一下 rep 重新 z-score 導致絕對強度被消掉。

## 整體 Set-Level 結果

Top phase-aware features：

```text
set_index_numeric                              rho =  0.4397
eccentric_pca_range_mean                       rho =  0.3377
eccentric_pca_range_last2                      rho =  0.3351
eccentric_wave_sim_to_first2_last_minus_first  rho = -0.3272
concentric_pca_range_mean                      rho =  0.3174
concentric_pca_range_last2                     rho =  0.3111
n_reps                                         rho = -0.2927
concentric_gyro_diff_rms_last2                 rho =  0.2830
eccentric_pca_movement_rate_mean               rho =  0.2801
eccentric_gyro_peak_abs_cv                     rho =  0.2740
```

解讀：

CE phase 特徵確實有 RPE 訊號，尤其是 phase waveform range、gyro 變化、以及和前兩下的相似度變化。但目前最強的單一訊號仍是第幾組。

## 假說檢查

針對原本提出的疲勞假說：

```text
phase_vector_similarity_slope                  rho = -0.2188
concentric_sec_last2_vs_first2                 rho =  0.1740
concentric_gyro_diff_rms_last2_vs_first2       rho =  0.1670
concentric_sec_slope                           rho =  0.1664
concentric_gyro_diff_rms_slope                 rho =  0.1294
ce_ratio_slope                                 rho =  0.1011
concentric_pca_movement_rate_slope             rho =  0.0663
concentric_pca_movement_rate_last2_vs_first2   rho =  0.0091
```

結論：

「向心變慢」有訊號，但不是最強訊號。比較被數據支持的是：

1. phase similarity 下降；
2. eccentric / concentric PCA range 增加；
3. concentric gyro diff RMS 增加；
4. 某些動作中向心時間變長。

「向心 movement rate 下降」目前沒有被整體數據強力支持。

## 每個動作的差異

每個動作最支持的 phase-aware fatigue feature 不同：

```text
db_shoulder_press:
concentric_gyro_diff_rms_last2_vs_first2 rho = 0.6275
concentric_gyro_diff_rms_slope           rho = 0.5933
phase_vector_similarity_slope            rho = -0.5848

db_triceps_curl:
concentric_gyro_diff_rms_last2_vs_first2 rho = 0.6535
concentric_gyro_diff_rms_slope           rho = 0.5251
concentric_sec_last2_vs_first2           rho = 0.3736

db_biceps_curl:
eccentric_wave_sim_to_first2_slope       rho = -0.4997
ce_ratio_slope                           rho = 0.4419
concentric_sec_slope                     rho = 0.4175

one_arm_db_row:
concentric_gyro_diff_rms_slope           rho = 0.4643
eccentric_wave_sim_to_first2_slope       rho = -0.3937
ce_ratio_slope                           rho = 0.3681
```

## 結論

數據支持「CE phase-aware fatigue」方向，但不支持一個過度簡化的規則，例如：

```text
向心越慢 => RPE 一定越高
```

更合理的規則是：

```text
疲勞狀態 = phase waveform shape/range 改變
        + concentric gyro 變化上升
        + phase similarity 下降
        + 部分動作向心時間拉長
        + set/rep progress 與累積 TUT
```

後續模型應該採用 exercise-aware phase fatigue score。也就是每個動作用不同權重看 CE phase 特徵，而不是所有動作共用同一條規則。
