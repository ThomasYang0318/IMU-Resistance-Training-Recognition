# Controlled One-Feature-at-a-Time Ablation for Borg/RPE

> Task ID: `009_controlled_one_feature_ablation`  
> Branch: `rep-segmentation-classification`  
> Task Owner: `Main Agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務回應一個關鍵資料設計問題：每個 exercise 的 RPE 都從 1 開始，因此 set index 與 cumulative TUT 很可能只是捕捉標註流程中的 within-exercise progression。為避免一次加入多個特徵造成過度解讀，本任務使用 leave-one-subject-out Ridge regression，固定 `exercise + set_index_numeric + cumulative_tut_exercise_sec` 為 progression control，每次只加入一個 workload、IMU trend 或 45 秒 delayed VO2 特徵。結果顯示，最佳 progression baseline 是 `exercise + cumulative_tut_exercise_sec`，RPE-only MAE 為 1.2206、VO2 overlap MAE 為 1.0579；沒有任何單一 IMU/VO2 特徵能超過此最佳 baseline。這代表目前資料最強證據是 within-exercise cumulative exposure，而不是單一 sensor feature 的獨立增益。

## Index Terms

IMU, Borg RPE, controlled ablation, one-feature model, cumulative TUT, delayed VO2, leave-one-subject-out

## I. Introduction

008 的 feature association 圖顯示 `cumulative_tut_exercise_sec` 與 `set_index_numeric` 和 RPE 有很強關聯。但使用者指出，每個動作的 RPE 都從 1 開始，這代表強相關可能只是因為標註規則與訓練流程：同一個 exercise 越後面的 set，RPE 自然越高。

因此本任務不再一次加入一整組 lowdim IMU 或 VO2 features，而是採用 one-feature-at-a-time ablation：

```text
固定同一個 progression baseline，
每次只新增一個 candidate feature，
看它是否還有額外預測增益。
```

## II. Task Definition

- Goal: 驗證在控制 exercise 與 within-exercise progression 後，單一 IMU / VO2 / workload feature 是否仍與 Borg/RPE 有額外關聯。
- Inputs:
  - `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
  - `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`
- Allowed Changes:
  - 新增 `tools/evaluate_controlled_one_feature_ablation.py`
  - 新增 `artifacts/fatigue_rpe_vo2/004_controlled_one_feature_ablation/`
  - 更新 `artifacts/fatigue_rpe_vo2/RESULTS_INDEX.md`
  - 更新 `docs/README.md`、`todo.md`
- Forbidden Changes:
  - 不改 schema
  - 不刪除或覆蓋既有正式結果
  - 不碰 `datasets/raw_data` dirty deletion 狀態
  - 不把 association 說成 causality
- Non-goals:
  - 不做大型 hyperparameter search
  - 不使用 high-dimensional phase features
  - 不宣稱 real-time fatigue prediction
- Outputs:
  - one-feature ablation table、model summary、fold metrics、prediction table、figures、summary JSON
- Acceptance Criteria:
  - 每個 candidate model 只能新增一個 numeric feature
  - baseline 和 candidate 使用同一個 LOSO split
  - 報告必須同時呈現 relative to full progression control 與 relative to best progression baseline
  - 明確標示是否有特徵超過最佳 progression baseline
- Dependencies:
  - 003 feature association evidence table
  - 021/022 feature datasets
- Handoff:
  - 下一步應改做 exercise-specific 或 interaction-based model，而不是繼續宣稱單一特徵強證據

## III. Input Data and Assumptions

資料：

| Dataset | Rows | Subjects | Purpose |
|---|---:|---:|---|
| `rpe_only_143_sets` | 143 | 6 | 檢查 workload 與 IMU trend |
| `vo2_lag45_96_sets` | 96 | 4 | 檢查同一批資料上的 workload、IMU trend 與 delayed VO2 |

衍生特徵：

```text
cumulative_tut_exercise_sec
```

定義為同一 `folder + exercise` 內，截至目前 set 的累積 `total_tut_sec`。這個欄位只在本 artifact 內衍生，不改原始資料 schema。

## IV. Method

所有 learned models 使用 Ridge regression，並固定使用 leave-one-subject-out。模型都包含 `exercise` one-hot 控制變項。

Progression baselines：

```text
M0: exercise mean
M1: exercise + set_index_numeric
M2: exercise + cumulative_tut_exercise_sec
M3: exercise + set_index_numeric + cumulative_tut_exercise_sec
```

One-feature candidate：

```text
M4(feature): M3 + exactly one candidate feature
```

主要比較：

```text
$\Delta MAE_{M3} = MAE_{M3} - MAE_{M4(feature)}$
```

另外加入更嚴格的比較：

```text
$\Delta MAE_{best} = MAE_{best\ progression\ baseline} - MAE_{M4(feature)}$
```

若 $\Delta MAE_{best} > 0$，才代表該單一 feature 超過最佳 progression baseline。

## V. Results

正式 artifact：

```text
artifacts/fatigue_rpe_vo2/004_controlled_one_feature_ablation/
```

### A. Progression Baselines

| Dataset | Model | MAE | Spearman $\rho$ | rounded $\pm1$ acc |
|---|---|---:|---:|---:|
| RPE-only 143 | M0 exercise mean | 1.3752 | 0.2704 | 0.6154 |
| RPE-only 143 | M1 exercise + set index | 1.2435 | 0.4940 | 0.6713 |
| RPE-only 143 | M2 exercise + cumulative TUT | **1.2206** | **0.5771** | 0.6643 |
| RPE-only 143 | M3 both progression controls | 1.2684 | 0.5349 | **0.6783** |
| VO2 lag45 96 | M0 exercise mean | 1.3727 | 0.0929 | 0.6042 |
| VO2 lag45 96 | M1 exercise + set index | 1.3051 | 0.2455 | 0.6667 |
| VO2 lag45 96 | M2 exercise + cumulative TUT | **1.0579** | **0.5381** | **0.7292** |
| VO2 lag45 96 | M3 both progression controls | 1.1187 | 0.4793 | 0.7083 |

重點：`exercise + cumulative_tut_exercise_sec` 是兩個 dataset 的最佳 progression baseline。把 `set_index_numeric` 和 cumulative TUT 同時放入 Ridge 後，MAE 反而略差，可能來自兩者高度重疊與小樣本不穩定。

### B. One-Feature Gains Relative to M3

相對於完整 progression control M3，部分特徵有小幅改善：

| Dataset | Added Feature | Family | $\Delta MAE_{M3}$ | $\Delta Spearman_{M3}$ |
|---|---|---|---:|---:|
| VO2 lag45 96 | `movement_rate_cv` | lowdim IMU | 0.0400 | 0.0338 |
| RPE-only 143 | `movement_rate_cv` | lowdim IMU | 0.0363 | 0.0000 |
| VO2 lag45 96 | `n_reps` | workload | 0.0353 | 0.0278 |
| VO2 lag45 96 | `sim_to_first_slope` | lowdim IMU | 0.0216 | 0.0276 |
| RPE-only 143 | `sim_to_first_slope` | lowdim IMU | 0.0140 | 0.0013 |

這些是 small/mixed gains，不能當作強證據。

### C. Strict Comparison Against Best Progression Baseline

更嚴格地和最佳 progression baseline M2 比較後：

| Dataset | Best Single Added Feature | Candidate MAE | Best Progression MAE | $\Delta MAE_{best}$ |
|---|---|---:|---:|---:|
| RPE-only 143 | `movement_rate_cv` | 1.2321 | 1.2206 | -0.0115 |
| VO2 lag45 96 | `movement_rate_cv` | 1.0788 | 1.0579 | -0.0208 |
| VO2 lag45 96 | `n_reps` | 1.0834 | 1.0579 | -0.0255 |
| VO2 lag45 96 | `sim_to_first_slope` | 1.0972 | 1.0579 | -0.0392 |

沒有任何單一 candidate feature 贏過最佳 progression baseline。VO2 features 也沒有超過 M2：

| VO2 Feature | Candidate MAE | $\Delta MAE_{best}$ | $\Delta Spearman_{best}$ |
|---|---:|---:|---:|
| `vo2_slope` | 1.1516 | -0.0936 | -0.0659 |
| `vo2_mean` | 1.1674 | -0.1094 | -0.1113 |
| `vo2_peak` | 1.1718 | -0.1138 | -0.1024 |

## VI. Figure and Table Reading Guide

主要檔案：

- `tables/controlled_one_feature_ablation.csv`：每列是一個 candidate feature，且只新增一個 feature。
- `metrics/model_summary.csv`：M0/M1/M2/M3 與所有 M4 candidate model 的 overall metrics。
- `metrics/fold_metrics.csv`：每個 held-out subject 的 fold metrics。
- `figures/progression_baselines_mae.png`：progression baselines 的 MAE 比較。
- `figures/one_feature_delta_mae.png`：相對 M3 的 one-feature $\Delta MAE$。
- `figures/one_feature_delta_vs_best_progression.png`：相對最佳 progression baseline 的嚴格 $\Delta MAE$。
- `figures/one_feature_coefficients.png`：單一新增 feature 的 standardized Ridge coefficient 方向。

讀圖重點：

- `one_feature_delta_mae.png` 大於 0 只代表該特徵比 M3 好，不代表它比所有 progression baseline 好。
- `one_feature_delta_vs_best_progression.png` 才是最嚴格判讀；本次所有 bars 都小於 0。
- `one_feature_coefficients.png` 的正負號只代表在 Ridge 模型中的方向，不代表因果。

## VII. Limitations

- Ridge 是線性模型；非線性或 exercise-specific interaction 可能仍有價值。
- 目前每個 exercise 的 RPE 從 1 開始，使 cumulative TUT / set order 有很強結構性優勢。
- VO2 結論只來自 96 sets、4 subjects。
- `cumulative_tut_exercise_sec` 使用 ground-truth set TUT；若未來使用 predicted segmentation，效果可能下降。
- 單一 feature 無法代表完整 sensor pattern；這輪只回答 one-feature evidence，不否定 group-level 或 interaction model。

## VIII. Reproducibility

重跑指令：

```bash
.venv311/bin/python tools/evaluate_controlled_one_feature_ablation.py \
  --output-dir artifacts/fatigue_rpe_vo2/004_controlled_one_feature_ablation
```

驗收指令：

```bash
.venv311/bin/python -m py_compile tools/evaluate_controlled_one_feature_ablation.py
python3 -m json.tool artifacts/fatigue_rpe_vo2/004_controlled_one_feature_ablation/summary.json
find artifacts/fatigue_rpe_vo2/004_controlled_one_feature_ablation -maxdepth 3 -type f | sort
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```

## IX. Conclusion

本任務完成 controlled one-feature-at-a-time ablation。結論是：在目前資料與標註設計下，RPE 主要被 within-exercise cumulative progression 解釋；單一 IMU 或 VO2 feature 加入後最多只有小幅改善，且沒有任何特徵超過最佳 progression baseline。因此下一步不應宣稱「某個單一 IMU/VO2 特徵能獨立證明疲勞」，而應改做 exercise-specific interaction、feature group ablation，或重新設計 RPE 標註以減少每個 exercise 重設為 1 的流程效應。

## References

- `artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table/summary.json`
- `artifacts/fatigue_rpe_vo2/004_controlled_one_feature_ablation/summary.json`
- `docs/tasks/008_feature_association_evidence_table_report.md`
