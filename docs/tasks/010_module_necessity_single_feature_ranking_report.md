# Module Necessity and Single-Feature Ranking for RPE/VO2

> Task ID: `010_module_necessity_single_feature_ranking`  
> Branch: `rep-segmentation-classification`  
> Task Owner: `Main Agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務把研究問題從「驗證某個假設模組重要」改成「開放式比較哪些單一特徵與模組最有證據」。分析同時包含兩層：第一層將所有 numeric features 做 single-feature ranking，檢查 raw / centered Spearman 與單特徵 LOSO Ridge MAE；第二層用 module ladder 比較 no exercise、exercise context、set index、cumulative TUT、lowdim IMU group 與 delayed VO2。結果顯示，目前和 RPE 最相關的單一特徵是 `cumulative_tut_exercise_sec`；exercise context 也明顯優於 global mean，支持動作切分/辨識的必要性。VO2 對目前 RPE label 的增益不強，但 019 顯示 GT-segmented IMU set features 可估計 VO2 mean，表示 VO2 更適合被定位為獨立的延遲生理負荷輸出，而不是目前 RPE 模型的必要輸入。

## Index Terms

IMU, Borg RPE, feature ranking, exercise segmentation, cumulative TUT, VO2 estimation, module ablation

## I. Introduction

使用者原本假設「動作切分」和「VO2 預估」可能重要，但也需要知道實際上哪個單一特徵最相關。因此本任務不只做模組驗證，也做開放式 single-feature ranking。這避免把分析變成只證明既有假設，而是讓資料回答：

```text
哪個單一 feature 最能解釋 RPE？
動作切分 / set-rep-TUT / IMU trend / VO2 哪些模組真的有證據？
```

## II. Task Definition

- Goal: 同時產生單一特徵排名與模組必要性證據，用於論文動機與系統架構說明。
- Inputs:
  - `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
  - `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`
  - `artifacts_rep_classification/019_vo2_gt_waveform_relation/summary.json`
- Allowed Changes:
  - 新增 `tools/analyze_module_necessity_rpe_vo2.py`
  - 新增 `artifacts/fatigue_rpe_vo2/005_module_necessity_rpe_vo2/`
  - 更新 `artifacts/fatigue_rpe_vo2/RESULTS_INDEX.md`
  - 更新 `docs/README.md`、`todo.md`
- Forbidden Changes:
  - 不改 schema
  - 不處理 raw data deletion dirty state
  - 不把 correlation 或 ablation 說成因果
  - 不預設 VO2 一定對 RPE 有用
- Non-goals:
  - 不做新的 VO2 原始重對齊
  - 不做大型模型搜尋
  - 不做 predicted segmentation gap
- Outputs:
  - single-feature ranking、module ladder metrics、module comparison table、VO2 estimability evidence、figures、summary JSON
- Acceptance Criteria:
  - numeric features 必須開放式排名，不只包含預設假設特徵
  - exercise context 以 categorical baseline 比較，不與 Spearman 硬比
  - VO2 對 RPE 與 VO2 作為獨立 target 必須分開解讀
- Dependencies:
  - 021 / 022 / 019 artifacts
- Handoff:
  - 下一步可做 exercise-specific feature group 或重新設計 RPE target。

## III. Input Data and Assumptions

資料：

| Dataset | Rows | Subjects | Purpose |
|---|---:|---:|---|
| RPE-only set table | 143 | 6 | single-feature ranking 與 RPE module ladder |
| RPE+VO2 lag45 subset | 96 | 4 | delayed VO2 與 RPE 的比較 |
| VO2 estimation summary 019 | 158 sets max | 6 | VO2 是否可由 GT-segmented IMU set features 估計 |

本任務延續 008/009 的重要修正：每個 exercise 的 RPE 都從 1 開始，因此 `set_index_numeric` 與 `cumulative_tut_exercise_sec` 反映的是 within-exercise progression，不應被說成直接生理疲勞。

## IV. Method

### A. Single-Feature Ranking

對所有 numeric candidate features 計算：

```text
raw Spearman
subject-centered Spearman
exercise-centered Spearman
subject+exercise-centered Spearman
single-feature LOSO Ridge MAE
```

`exercise_context` 是 categorical context，因此不計 Spearman；它用 exercise mean 的 LOSO MAE 與 global mean baseline 比較。

### B. Module Ladder

RPE 模組階梯：

```text
M0: global mean
M1: exercise mean
M2: set_index_numeric without exercise
M3: exercise + set_index_numeric
M4: exercise + cumulative_tut_exercise_sec
M5: M4 + lowdim IMU trend group
M6: M5 + delayed VO2, only for lag45 subset
```

主要比較使用：

```text
$\Delta MAE = MAE_{baseline} - MAE_{augmented}$
```

正值代表 augmented module 降低錯誤。

### C. VO2 Estimability

VO2 的必要性分成兩個命題：

1. delayed VO2 是否改善目前 RPE label；
2. VO2 是否可以作為獨立生理負荷 target 被 IMU set features 估計。

第二點引用 019 的 VO2 estimation result，不混入 RPE 結論。

## V. Results

正式 artifact：

```text
artifacts/fatigue_rpe_vo2/005_module_necessity_rpe_vo2/
```

### A. Top Single Features by Raw Association

RPE-only 143 sets：

| Rank | Feature | Family | Raw $\rho$ | Single-feature MAE |
|---:|---|---|---:|---:|
| 1 | `cumulative_tut_exercise_sec` | cumulative exposure | 0.4987 | 1.3548 |
| 2 | `set_index_numeric` | progression | 0.4397 | 1.4323 |
| 3 | `n_reps` | workload/TUT | -0.2927 | 1.5006 |
| 4 | `pca_diff_rms_max` | PCA waveform | -0.2873 | 1.5331 |
| 5 | `pca_diff_rms_mean` | PCA waveform | -0.2861 | 1.4840 |

VO2 overlap lag45 96 sets：

| Rank | Feature | Family | Raw $\rho$ | Single-feature MAE |
|---:|---|---|---:|---:|
| 1 | `cumulative_tut_exercise_sec` | cumulative exposure | 0.5760 | 1.1332 |
| 2 | `set_index_numeric` | progression | 0.4144 | 1.3482 |
| 3 | `movement_rate_cv` | variability | -0.3943 | 1.3556 |
| 4 | `vo2_slope` | delayed VO2 | 0.3639 | 1.4568 |
| 5 | `concentric_gain_last2_vs_first2` | phase timing | 0.3621 | 1.5943 |

解讀：若只問「哪個單一 numeric feature 最相關」，答案是 `cumulative_tut_exercise_sec`。VO2 中最有 RPE association 的是 `vo2_slope`，但它不是整體最強特徵，且單特徵 LOSO MAE 不佳。

### B. Top Single Features by LOSO MAE

RPE-only 143 sets：

| Feature | Type | MAE | Spearman |
|---|---|---:|---:|
| `cumulative_tut_exercise_sec` | numeric | 1.3548 | 0.3966 |
| `exercise_context` | categorical | 1.3752 | 0.2704 |
| `set_index_numeric` | numeric | 1.4323 | 0.2994 |

VO2 overlap lag45 96 sets：

| Feature | Type | MAE | Spearman |
|---|---|---:|---:|
| `cumulative_tut_exercise_sec` | numeric | 1.1332 | 0.4507 |
| `set_index_numeric` | numeric | 1.3482 | 0.1622 |
| `sim_to_first_slope` | numeric | 1.3528 | 0.0443 |
| `movement_rate_cv` | numeric | 1.3556 | -0.0760 |
| `exercise_context` | categorical | 1.3727 | 0.0929 |

### C. Module Ladder Evidence

| Dataset | Comparison | MAE reduction | Spearman gain | Evidence |
|---|---|---:|---:|---|
| RPE-only | exercise vs global mean | 0.1754 | 0.7087 | strong |
| RPE-only | exercise + set index vs set index alone | 0.1888 | 0.1946 | strong |
| RPE-only | cumulative TUT vs set index | 0.0229 | 0.0831 | mixed/small |
| RPE-only | lowdim IMU group after TUT | 0.0110 | -0.0115 | not supported |
| VO2 lag45 | exercise vs global mean | 0.0720 | 0.5496 | mixed/small |
| VO2 lag45 | cumulative TUT vs set index | 0.2472 | 0.2926 | strong |
| VO2 lag45 | lowdim IMU group after TUT | -0.0561 | -0.0017 | not supported |
| VO2 lag45 | delayed VO2 after IMU | -0.1255 | -0.0972 | not supported |

### D. VO2 Estimability as an Independent Target

019 顯示 GT-segmented IMU set features 可以估計 VO2 mean：

| Target | Lag | Model | Sets | Subjects | MAE | $R^2$ | Spearman |
|---|---:|---|---:|---:|---:|---:|---:|
| `vo2_mean` | 0 s | random forest | 158 | 6 | 2.1678 | 0.3438 | 0.6431 |
| `vo2_mean` | 10 s | random forest | 158 | 6 | 2.3230 | 0.3670 | 0.6798 |
| `vo2_mean` | 10 s | ridge | 158 | 6 | 2.4234 | 0.1211 | 0.6403 |

這支持「估計 VO2」作為生理負荷輸出的研究方向；但目前不支持「VO2 是 RPE 預測必要特徵」。

## VI. Figure and Table Reading Guide

主要檔案：

- `tables/single_feature_open_ranking.csv`：所有 numeric features 與 `exercise_context` 的開放式 ranking。
- `metrics/rpe_module_ladder.csv`：RPE 模組階梯的 LOSO metrics。
- `tables/module_necessity_comparisons.csv`：模組間 $\Delta MAE$ 與 $\Delta Spearman$。
- `tables/vo2_estimability_evidence.csv`：019 VO2 estimation 的摘要證據。
- `figures/single_feature_raw_spearman_ranking.png`：各 dataset top numeric features 的 raw Spearman。
- `figures/rpe_module_ladder_mae.png`：RPE module ladder MAE。
- `figures/module_comparison_delta_mae.png`：模組增益比較。
- `figures/vo2_estimability_spearman.png`：VO2 mean 可估計性。

讀圖重點：

- `single_feature_raw_spearman_ranking.png` 回答「單一 numeric feature 誰最相關」。
- `rpe_module_ladder_mae.png` 回答「加入哪個模組讓 RPE MAE 降低」。
- `module_comparison_delta_mae.png` 大於 0 才表示新增模組改善。
- `vo2_estimability_spearman.png` 是 VO2 作為 target 的證據，不是 VO2 對 RPE 的證據。

## VII. Limitations

- 每個 exercise 的 RPE 從 1 開始，使 cumulative TUT 和 set index 有結構性優勢。
- VO2 overlap 只有 96 sets、4 subjects。
- Module ladder 使用 Ridge；非線性 exercise-specific 模型可能有不同結果。
- Single-feature ranking 不代表因果。
- 019 的 VO2 estimability 使用 GT segmentation，若改成 predicted segmentation 仍需重新驗證。

## VIII. Reproducibility

重跑指令：

```bash
.venv311/bin/python tools/analyze_module_necessity_rpe_vo2.py \
  --output-dir artifacts/fatigue_rpe_vo2/005_module_necessity_rpe_vo2
```

驗收指令：

```bash
.venv311/bin/python -m py_compile tools/analyze_module_necessity_rpe_vo2.py
python3 -m json.tool artifacts/fatigue_rpe_vo2/005_module_necessity_rpe_vo2/summary.json
find artifacts/fatigue_rpe_vo2/005_module_necessity_rpe_vo2 -maxdepth 3 -type f | sort
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```

## IX. Conclusion

本任務完成 single-feature ranking 與 module necessity ablation。最可支撐的主張是：目前 RPE 最主要反映同一 exercise 內的累積進程，因此動作切分/辨識與 set/rep/TUT segmentation 是必要的架構前提；最強單一 numeric feature 是 cumulative TUT。VO2 的合理定位不是「目前 RPE 預測必要特徵」，而是可由 IMU set features 估計的延遲生理負荷 target，可作為系統的第二輸出或輔助解釋訊號。

## References

- `artifacts/fatigue_rpe_vo2/005_module_necessity_rpe_vo2/summary.json`
- `artifacts_rep_classification/019_vo2_gt_waveform_relation/summary.json`
- `docs/tasks/009_controlled_one_feature_ablation_report.md`
