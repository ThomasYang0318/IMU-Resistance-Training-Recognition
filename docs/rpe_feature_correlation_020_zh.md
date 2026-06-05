# 020 RPE 特徵相關度分析

> 2026-05-17 artifact cleanup note：020 舊 artifacts 已瘦身刪除；目前保留 021 的 set-level feature dataset 作為 022 的追溯輸入。本文件保留作歷史分析脈絡。

## 目的

這版先不訓練新模型，而是回答一個更前置的問題：

> 哪些 IMU / TUT / 疲勞趨勢特徵真的和 Borg/RPE 有關？

為了避免自動 rep 切割誤差干擾判斷，本次沿用第 018 版的 ground-truth rep / phase segmentation dataset。`yushuan0513workout` 的 RPE 標籤只有 `1`，無法提供有效相關度，因此採用第 018 版的 `exclude_sparse` dataset。

## 輸入與輸出

輸入：

```text
artifacts_rep_classification/018_borg_gt_waveform_relation_exclude_sparse/018_gt_rep_waveform_borg_dataset.csv
```

輸出：

```text
artifacts_rep_classification/020_rpe_feature_correlation_analysis/
```

主要輸出檔：

- `020_rpe_rep_level_correlations.csv`
- `020_rpe_set_level_correlations.csv`
- `020_rpe_rep_top_raw_spearman.png`
- `020_rpe_rep_top_subject_exercise_centered_spearman.png`
- `020_rpe_set_top_raw_spearman.png`
- `020_rpe_set_top_subject_exercise_centered_spearman.png`
- `020_rpe_rep_feature_group_summary.png`
- `020_rpe_set_feature_group_summary.png`

## 分析層級

### Rep-level

每一下 rep 對應該 rep 的 Borg/RPE 標籤。這會回答：

> 在一組動作進行到某一下時，哪些特徵和當下 RPE 上升最相關？

樣本數：

```text
1396 reps
```

### Set-level

每一組 set 取最後一個 Borg/RPE 作為該組的主觀疲勞目標。這會回答：

> 整組動作完成後，哪些整組趨勢和 final RPE 最相關？

樣本數：

```text
119 sets
```

## 相關度指標

主要看 `Spearman correlation`，因為 Borg/RPE 是等級量表，不應只用線性 Pearson 解讀。

表格同時輸出：

- `raw_spearman`：直接把所有人所有動作合在一起算；
- `exercise_centered_spearman`：扣掉動作種類平均差異；
- `subject_centered_spearman`：扣掉受試者主觀尺度差異；
- `subject_exercise_centered_spearman`：扣掉「同一人、同一動作」的 baseline，較接近組內疲勞趨勢。

## 主要結果

### Rep-level Top Raw Spearman

```text
rep_progress              rho =  0.5166
set_index_numeric         rho =  0.5082
rep_index / rep_order     rho =  0.4960 ~ 0.4961
cumulative_tut_sec        rho =  0.4254
cumulative_eccentric_sec  rho =  0.4247
cumulative_concentric_sec rho =  0.4005
kg_x_rep                  rho =  0.3892
kg_x_cumulative_tut       rho =  0.3542
```

解讀：

RPE 最明顯跟「做到第幾下」、「第幾組」、「累積 TUT」有關。這合理，因為 Borg/RPE 本身就是累積疲勞感，而不是單一瞬間波形振幅。

### Rep-level 波形 / 疲勞特徵

排除進度、重量、TUT 後，較明顯的特徵是：

```text
my_diff_abs_mean              rho = -0.1639
my_diff_rms                   rho = -0.1638
pca_diff_abs_mean             rho = -0.1529
pca_diff_rms                  rho = -0.1520
pca_range                     rho = -0.1379
gyro_diff_change_from_first2  rho =  0.1152
velocity_loss_proxy           rho =  0.0926
```

解讀：

單一下 rep 的波形本身和 RPE 的 raw correlation 只有弱相關。比較有意義的是「相對前兩下的變化」，例如 gyro diff 上升、velocity proxy 下降，但目前強度還不夠作為單獨 RPE predictor。

### Set-level Top Raw Spearman

```text
set_index_numeric                 rho =  0.4435
n_reps                            rho = -0.2700
pca_diff_rms_mean                 rho = -0.2483
pca_diff_rms_max                  rho = -0.2481
gyro_diff_gain_last2_vs_first2    rho =  0.2302
gyro_mag_diff_rms_slope           rho =  0.1878
duration_gain_last2_vs_first2     rho =  0.1321
velocity_loss_last2_vs_first2     rho =  0.1312
```

解讀：

set-level 裡面，`gyro_diff_gain_last2_vs_first2`、`velocity_loss_last2_vs_first2`、`duration_gain_last2_vs_first2` 比單一下波形更符合疲勞假設。這表示後續 RPE 模型應該以整組趨勢為主，而不是逐 rep 瞬間分類。

## 結論

目前資料支持三個方向：

1. RPE 最強訊號是累積進度：第幾組、第幾下、累積 TUT。
2. 波形本身有訊號，但 raw correlation 偏弱，不能單獨撐起高準度 RPE。
3. 比起單一下的絕對波形，`last2 vs first2` 的變化更有意義，尤其是 gyro 變化、速度下降、rep duration 增加。

因此下一版若要預測 RPE，建議不要只做 rep-level regression，而是改成：

```text
set-level fatigue trend model
= progress + cumulative TUT + kg + exercise
  + velocity loss
  + gyro change last2 vs first2
  + waveform similarity decay
  + few-shot subject calibration
```

## 目前限制

- Borg/RPE 是主觀尺度，不同受試者的 `7` 不一定等價；
- `kg` 不是 relative load，缺少 1RM 或個人最大能力；
- 目前第 018 版波形特徵多數是 rep 內 z-score 後的形狀特徵，絕對強度訊號被弱化；
- set-level 只有 119 組，若要穩定建模需要更多受試者或更多 session。
