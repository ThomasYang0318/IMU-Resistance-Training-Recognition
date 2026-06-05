# Low-Dimensional Set-Level IMU Trend 與 Delayed VO2 正式驗證

> Task ID: `007_lowdim_set_trend_vo2_validation`  
> Branch: `rep-segmentation-classification`  
> Task Owner: `Main Agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務回應 006 的結論：高維 CE phase features 在 LOSO 下容易過擬合，因此改用低維、非 phase-specific 的 set-level trend features 驗證 IMU 與 delayed VO2 對 Borg/RPE 的增益。Model A 使用 workload/dose features (`kg`, `n_reps`, `total_tut_sec`)，另以 `set_index_numeric` 作 diagnostic；Model B 加入 6 個低維 IMU trend features；Model C 再加入 raw delayed VO2 (`vo2_mean`, `vo2_peak`, `vo2_slope`)。結果顯示，在 96-set VO2 overlap subset 上，low-dimensional IMU trend 明顯改善 workload baseline，45 秒 VO2 lag 的 random forest 進一步達到 MAE 1.2300、Spearman $\rho$ 0.2797、rounded $\pm1$ accuracy 0.6875。但在 143-set RPE-only 全量上，set order diagnostic 與 exercise baseline 仍更穩，表示低維 non-phase 特徵有研究價值，但需要更多 subjects 與更嚴格的 real-time protocol。

## Index Terms

IMU, set-level trend, Borg RPE, VO2, resistance training, non-phase features, leave-one-subject-out

## I. Introduction

006 正式驗證發現，直接加入 105 個 GT phase-aware IMU features 會降低跨人泛化。因此本任務將第一版系統主線收斂為「set-level IMU trend + delayed VO2」，並將 phase 保留為解釋性分析，而不是第一版必要預測模組。

本任務的問題是：

```text
不用 CE phase-specific features，只用低維 set-level IMU trend，
是否能比 workload/TUT baseline 更好地預測 Borg/RPE？
delayed VO2 是否能再提供額外增益？
```

## II. Task Definition

- Goal: 以低維 non-phase features 正式驗證 set-level IMU trend 與 delayed VO2 對 Borg/RPE 的增益。
- Inputs:
  - `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
  - `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`
- Allowed Changes:
  - 新增 `tools/evaluate_lowdim_set_trend_vo2.py`
  - 新增 `artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval/`
  - 更新 `artifacts/fatigue_rpe_vo2/RESULTS_INDEX.md`
  - 更新 `docs/README.md`、`todo.md`
- Forbidden Changes:
  - 不使用 CE phase-specific 欄位作主模型
  - 不使用 subject-normalized VO2 欄位作主模型
  - 不刪除、回復或搬移既有 artifacts
  - 不處理 `datasets/raw_data` dirty deletion 狀態
- Non-goals:
  - 不做 predicted segmentation gap
  - 不做大型模型或超參數搜尋
  - 不宣稱此結果是逐 rep real-time prediction
- Outputs:
  - formal artifact、metrics CSV、prediction CSV、figure PNG、summary JSON、task report
- Acceptance Criteria:
  - 使用 leave-one-subject-out
  - 明確分開 workload/dose、set-order diagnostic、lowdim IMU trend、delayed VO2
  - 報告 Model B-A 與 Model C-B 的增益
- Dependencies:
  - 021 RPE set-level feature table
  - 022 RPE+VO2 merged set table
- Handoff:
  - 下一任務可做 real-time prefix features 或 predicted segmentation gap

## III. Input Data and Assumptions

資料：

| Dataset | Rows | Unique Sets | Subjects | Notes |
|---|---:|---:|---:|---|
| 021 RPE set-level table | 143 | 143 | 6 | 無 VO2，包含 lowdim IMU set features |
| 022 RPE+VO2 merged table | 572 | 96 | 4 | 每個 lag 一列；60 秒 lag 為 92 rows |

021 與 022 的非 VO2 set-level 欄位完全對齊。022 只覆蓋 021 中的 96 個 common sets；因此 Model C 的 VO2 結論只能代表 overlap subset。

## IV. Method

### A. Feature Groups

所有 learned models 均加入 `exercise` one-hot 作為類別控制變項；baseline 則使用 training subjects 的 exercise mean。

Model A workload/dose:

```text
kg
n_reps
total_tut_sec
```

Set-order diagnostic:

```text
set_index_numeric
```

這個欄位不是主模型結論，因為它高度代表固定實驗流程順序與累積疲勞 proxy。

Model B low-dimensional IMU set trend:

```text
rep_duration_cv
movement_rate_cv
gyro_diff_gain_last2_vs_first2
gyro_mag_diff_rms_slope
sim_to_first_slope
pca_diff_rms_mean
```

Model C delayed VO2:

```text
vo2_mean
vo2_peak
vo2_slope
```

排除欄位：

- CE phase-specific：`concentric_*`, `eccentric_*`, `mean_concentric_*`, `mean_eccentric_*`, `ce_ratio_*`。
- Subject-normalized VO2：`vo2_mean_z_subject`, `vo2_mean_delta_subject_min`, `vo2_peak_z_subject`, `vo2_peak_delta_subject_min`。
- VO2 interaction：`vo2_mean_x_total_tut`, `vo2_mean_x_n_reps` 等，避免混淆解釋。

### B. Validation

使用 leave-one-subject-out。模型包含 ridge regression 與 random forest regression。每個 VO2 lag 獨立評估，避免同一 set 的多個 lag row 在同一模型中互相污染。

主要指標：

```text
MAE, RMSE, R2, Spearman rho, rounded exact accuracy, rounded +/-1 accuracy
```

## V. Results

正式 artifact：

```text
artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval/
```

### A. RPE-Only 143 Sets

| Model | MAE | Spearman $\rho$ | rounded $\pm1$ acc |
|---|---:|---:|---:|
| Baseline exercise mean | 1.3752 | 0.2704 | 0.6154 |
| A workload/dose ridge | 1.5060 | 0.2087 | 0.6014 |
| A workload/dose RF | 1.6474 | 0.0075 | 0.4895 |
| A + set order diagnostic RF | **1.3595** | 0.3964 | 0.6154 |
| B lowdim set trend ridge | 1.5706 | 0.2572 | 0.5524 |
| B lowdim set trend RF | 1.5852 | 0.0600 | 0.5455 |

解讀：在 143-set 全量 RPE-only 條件下，低維 IMU trend 沒有贏過 exercise mean baseline，也沒有贏過 set-order diagnostic。這表示若沒有 VO2 overlap subset 或更強 subject calibration，目前低維 IMU trend 還不足以單獨支撐全量跨人 RPE 預測。

### B. RPE+VO2 Overlap 96 Sets

在 96-set overlap subset 上，結果較有利於低維特徵：

| Lag | Model | MAE | Spearman $\rho$ | rounded $\pm1$ acc |
|---:|---|---:|---:|---:|
| 45 s | C lowdim + VO2 RF | **1.2300** | 0.2797 | **0.6875** |
| 45 s | B lowdim RF | 1.3207 | 0.1710 | 0.6354 |
| 45 s | A + set order diagnostic RF | 1.2938 | 0.2452 | 0.6250 |
| 60 s | B lowdim ridge | 1.2786 | **0.3790** | 0.6413 |

45 秒 lag 下，VO2 對 random forest 的增益為：

```text
C - B MAE reduction = 0.0906
Spearman gain = 0.1087
rounded +/-1 acc gain = 0.0521
```

在 96-set overlap 上，B lowdim 對 workload/dose baseline 的增益也明顯，例如 random forest 在 0 到 45 秒 lag 的 `B - A` MAE reduction 為 0.2271。

### C. Interpretation

本次結果支持三個判斷：

1. 低維 non-phase IMU trend 比高維 phase feature 更適合作第一版模型，但它的有效性目前主要出現在 96-set VO2 overlap subset。
2. `set_index_numeric` 仍是很強的 diagnostic proxy，代表實驗順序/累積疲勞在資料中影響很大；論文應避免把它當成純 IMU feature。
3. delayed VO2 在 45 秒 lag 與 lowdim IMU trend 結合時有目前最好的 MAE，但樣本只有 4 subjects、96 sets，因此只能說是 promising，不是強結論。

## VI. Figure and Table Reading Guide

主要檔案：

- `metrics/nested_model_summary.csv`：所有 lowdim nested model 的 LOSO overall metrics。
- `metrics/model_delta_summary.csv`：order diagnostic、lowdim IMU、VO2 的增益。
- `metrics/fold_metrics.csv`：每個 held-out subject 的 fold metrics。
- `tables/nested_model_predictions.csv`：所有模型的 out-of-subject predictions。
- `figures/rpe_lowdim_nested_mae.png`：143-set RPE-only lowdim MAE。
- `figures/vo2_lowdim_lag_metrics.png`：每個 VO2 lag 的 lowdim MAE/Spearman。
- `figures/lowdim_incremental_mae_gain.png`：lowdim IMU 與 VO2 的 MAE reduction。
- `figures/best_lowdim_predictions.png`：最佳 lowdim model 的 prediction scatter。

圖表解讀：

- `rpe_lowdim_nested_mae.png` 用來看 lowdim set trend 是否比 workload/dose 或 set-order diagnostic 更好。
- `lowdim_incremental_mae_gain.png` 左圖大於 0 表示 lowdim IMU trend 相對 workload/TUT 有增益；右圖大於 0 表示 delayed VO2 相對 lowdim IMU trend 有增益。
- `vo2_lowdim_lag_metrics.png` 可讀出 VO2 lag 是否集中在某個延遲時間有用；本次最明顯是 45 秒。

## VII. Limitations

- RPE+VO2 結論只來自 4 subjects、96 sets。
- 低維 set-trend features 仍是 post-set summary，不是每一下 rep 中即時可用。
- `set_index_numeric` 不能被當作生理或 IMU 特徵，只能作資料流程 diagnostic。
- 本次排除 phase-specific features，不代表 phase 無研究價值；phase 仍適合放在解釋與動作別分析。
- 尚未評估 predicted segmentation 對這些低維 features 的影響。

## VIII. Reproducibility

重跑指令：

```bash
.venv311/bin/python tools/evaluate_lowdim_set_trend_vo2.py \
  --output-dir artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval
```

驗收指令：

```bash
.venv311/bin/python -m py_compile tools/evaluate_lowdim_set_trend_vo2.py
python3 -m json.tool artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval/summary.json
find artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval -maxdepth 3 -type f | sort
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```

若最後一行只命中各文件中的檢查指令本身，代表沒有新增未決標記。

## IX. Conclusion

本任務支持將第一版主線改為 low-dimensional non-phase set-level trend，而不是 CE phase high-dimensional model。最有希望的組合是 lowdim IMU trend + delayed VO2 45 秒 lag；但全量 RPE-only 上仍沒有勝過 set-order diagnostic 或 exercise mean baseline。因此論文主張應保守：低維 set-level IMU trend 與 delayed VO2 可作為 Borg/RPE 相關的候選特徵，但仍需更多 subjects、real-time prefix features 與 predicted segmentation gap 驗證。

## References

- `docs/tasks/006_gt_phase_imu_vo2_rpe_formal_validation_report.md`
- `artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval/summary.json`
- `artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval/summary.json`
