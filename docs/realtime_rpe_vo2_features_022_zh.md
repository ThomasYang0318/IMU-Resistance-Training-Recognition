# 022 即時 RPE 特徵與 VO2 融合分析

## 目的

這版回答：

> 如果未來要用即時波形估 RPE，並把 VO2 即時值也納入考慮，應該抓哪些特徵？

本版使用第 021 版補上 `yushuan0513workout` 後的 set-level RPE 特徵，並合併第 019 版 VO2 lag window。VO2 是呼吸代謝訊號，會落後動作，因此不只看當下，也看 set 後 `0/10/20/30/45/60s`。

## 輸入

```text
RPE set features:
artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv

VO2 set features:
artifacts_rep_classification/019_vo2_gt_waveform_relation/019_vo2_set_waveform_dataset.csv
```

## 輸出

```text
artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/
```

主要檔案：

- `022_realtime_rpe_vo2_merged_set_dataset.csv`
- `022_realtime_rpe_vo2_feature_correlations.csv`
- `022_top_realtime_features_by_lag_raw_spearman.png`
- `022_top_vo2_features_by_lag_raw_spearman.png`
- `022_vo2_feature_correlation_by_lag.png`

## 可用資料

只有同時具備 RPE workbook 與 VO2 對齊的受試者能進入本分析。

```text
subjects = haoyu, yanz, yoru, yushuan
sets = 96
lag rows = 572
```

`kevin`、`ziho` 有 VO2 但沒有 RPE workbook；`hsianshun`、`tsenyu` 有 RPE 但沒有 VO2；`thomas` 的 VO2 與 IMU 日期不重疊。

## 主要結果

### 最穩定的即時特徵

跨 lag 來看，最穩定的特徵是：

```text
set_index_numeric               rho ~= 0.41
movement_rate_cv                rho ~= -0.39 to -0.40
concentric_gain_last2_vs_first2 rho ~= 0.35 to 0.36
gyro_mag_diff_rms_slope         rho ~= 0.36
sim_to_first_slope              rho ~= -0.32
concentric_sec_slope            rho ~= 0.30 to 0.32
rep_duration_cv                 rho ~= -0.31
```

解讀：

RPE 仍然最受「第幾組」影響。純 IMU 中比較值得即時抓的是：

- 向心時間是否逐漸變長；
- gyro 變化是否上升；
- movement rate 是否變得不穩；
- 和前面 rep 的相似度是否下降；
- rep duration / concentric duration 的 slope。

### VO2 特徵

VO2 的 raw correlation 有訊號，但不是最穩定主訊號：

```text
lag 10s:
vo2_mean_delta_subject_min rho = -0.3500
vo2_mean_x_n_reps          rho = -0.3158
vo2_peak_delta_subject_min rho = -0.3108
vo2_mean                   rho = -0.2868
vo2_peak                   rho = -0.2827

lag 45s:
vo2_slope                  rho =  0.3639
```

注意：VO2 和 RPE 在這批資料中常出現負相關，代表 raw VO2 受到休息、動作種類、受試者基準、呼吸延遲影響很大。VO2 應當作輔助生理負荷，不應單獨拿來預測 RPE。

## 即時系統建議

### 每個 rep 完成後更新

即時 IMU 應抓：

- `rep_duration_sec`
- `concentric_sec`
- `eccentric_sec`
- `concentric_ratio`
- `pca_range`
- `gyro_mag_diff_rms`
- `movement_rate = pca_range / rep_duration`
- `sim_to_first`
- `sim_to_prev`

然後在線上維護：

- `rep_progress`
- `cumulative_tut_sec`
- `duration_slope`
- `concentric_sec_slope`
- `gyro_mag_diff_rms_slope`
- `velocity_loss_last2_vs_first2`
- `concentric_gain_last2_vs_first2`
- `sim_to_first_slope`
- `movement_rate_cv`

### VO2 即時值

VO2 應抓：

- `vo2_mean_10s`
- `vo2_peak_10s`
- `vo2_delta_from_subject_min_10s`
- `vo2_slope_45s`
- `vo2_mean_x_n_reps`

如果要即時顯示，不建議把 VO2 當成當下 rep 的標籤；比較合理是顯示成：

```text
IMU fatigue state = 每一下更新
VO2 physiological load = 延遲 10-60 秒更新
RPE estimate = IMU fatigue state + VO2 delayed load + 個人校正
```

## 結論

即時 RPE 預估的核心應該是 IMU set-level fatigue state。VO2 有幫助，但目前資料顯示它更像延遲輔助訊號，不是主要預測來源。後續若要提升 VO2 的可用性，應改用 subject baseline-normalized VO2、VO2 AUC、rest-adjusted VO2，而不是 raw VO2 instant value。
