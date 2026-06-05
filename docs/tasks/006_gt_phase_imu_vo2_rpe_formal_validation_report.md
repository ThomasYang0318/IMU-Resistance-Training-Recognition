# GT Phase-Aware IMU, Delayed VO2 與 Borg/RPE 正式驗證

> Task ID: `006_gt_phase_imu_vo2_rpe_formal_validation`  
> Branch: `rep-segmentation-classification`  
> Task Owner: `Main Agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務將 005 研究框架轉成第一個 formal validation experiment，使用既有 ground-truth CE phase set features 與 delayed VO2 merged dataset，以 leave-one-subject-out 驗證 IMU/VO2 feature groups 是否能提升 Borg/RPE 預測。結果顯示，在 143 個 RPE sets 上，metadata/progress random forest 是最穩定模型，MAE 為 1.2456，Spearman $\rho$ 為 0.4770；直接加入 105 個高維 IMU phase features 反而降低跨人泛化。在 96 個 RPE+VO2 overlap sets 上，45 秒 lag 的 VO2 random forest 對 MAE 有小幅改善，但 Spearman 仍偏低，表示 delayed VO2 目前只能作為弱輔助訊號。

## Index Terms

IMU, VO2, Borg RPE, resistance training, fatigue-related movement change, leave-one-subject-out, nested model comparison

## I. Introduction

005 任務已將研究定位整理為：IMU 量化 fatigue-related movement changes，RPE/Borg 提供主觀 exertion label，VO2 提供 delayed physiological-load covariate。本任務進一步用正式 artifact 驗證這個框架中的 nested model 假說：

```text
Model A: metadata + progress
Model B: Model A + GT phase-aware IMU features
Model C: Model B + delayed VO2 features
Model D: Model C + few-shot subject calibration
```

這次不是重新抽特徵，而是使用已保留的 023 與 022 結果表，先回答「目前資料規模下，加入 IMU phase features 或 VO2 是否真的改善跨人 RPE 預測」。

## II. Task Definition

- Goal: 跑一個可重複的 formal validation experiment，量化 IMU phase features 與 delayed VO2 對 Borg/RPE 預測的增益。
- Inputs:
  - `artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/023_phase_aware_set_feature_dataset.csv`
  - `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`
  - `docs/tasks/005_research_framework_imu_vo2_rpe_report.md`
- Allowed Changes:
  - 新增 `tools/evaluate_imu_vo2_rpe_framework.py`
  - 新增 `artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval/`
  - 新增 `artifacts/fatigue_rpe_vo2/RESULTS_INDEX.md`
  - 更新 `docs/README.md`、`README.md`、`todo.md` 的短索引或任務狀態
- Forbidden Changes:
  - 不刪除或回復既有 artifacts
  - 不處理 `datasets/raw_data` 的 dirty deletion 狀態
  - 不修改 schema、artifact root 或核心評估指標
  - 不把 subject-normalized VO2 欄位放入 primary LOSO 模型，以避免 held-out subject 統計洩漏
- Non-goals:
  - 不評估 predicted rep/phase segmentation deployment gap
  - 不做大型超參數搜尋
  - 不導入 RAG/MCP 或新資料庫
- Outputs:
  - formal artifact、metrics CSV、prediction CSV、figure PNG、summary JSON、task report
- Acceptance Criteria:
  - 使用 leave-one-subject-out
  - 輸出 nested model metrics
  - 明確回報 Model B-A 與 Model C-B 的增益
  - `summary.json` 可被 JSON parser 讀取
- Dependencies:
  - 023 ground-truth phase-aware set features
  - 022 RPE+VO2 lag merged set dataset
- Handoff:
  - 下一步可做低維 feature selection 或 predicted segmentation gap experiment

## III. Input Data and Assumptions

資料盤點結果：

| Dataset | Rows | Unique Sets | Subjects | Notes |
|---|---:|---:|---:|---|
| 023 phase-aware RPE set table | 143 | 143 | 6 | 目標欄位為 `borg` |
| 022 RPE+VO2 merged table | 572 | 96 | 4 | `lag_sec = 0, 10, 20, 30, 45, 60`; 60 秒 lag 有 92 rows |

Join key 使用：

```text
folder + exercise + set_id
```

VO2 comparison 採 per-lag evaluation，也就是每個 `lag_sec` 各自形成一個 96-set 或 92-set dataset，避免把同一 set 的多個 lag row 在同一模型中當成獨立樣本。

## IV. Method

### A. Model Stages

Baseline:

```text
exercise mean from training subjects
```

Model A:

```text
exercise one-hot + set_index_numeric + kg + n_reps
```

Model B:

```text
Model A + all numeric GT phase-aware IMU set features
```

Model C:

```text
Model B + raw delayed VO2 features
```

Primary Model C 排除 `vo2_mean_z_subject`、`vo2_mean_delta_subject_min`、`vo2_peak_z_subject`、`vo2_peak_delta_subject_min`，因為這些 subject-normalized 欄位可能使用 held-out subject 的全資料統計。

Model D:

```text
Model C prediction + first-set-per-exercise subject calibration offset
```

Model D 是補充分析，不與 Model A/B/C 做完全等量比較，因為它使用 held-out subject 的少量標籤做 calibration，且評估時排除 calibration rows。

### B. Validation

主要驗證使用 leave-one-subject-out。每個 fold 用一個 `folder` 作為 held-out subject，其餘 subjects 訓練。

模型：

- Ridge regression with median imputation and standard scaling
- Random forest regression with median imputation

主要指標：

```text
MAE, RMSE, R2, Spearman rho, rounded exact accuracy, rounded +/-1 accuracy
```

增益定義：

```text
MAE reduction = MAE(previous stage) - MAE(next stage)
Spearman gain = Spearman(next stage) - Spearman(previous stage)
```

正的 MAE reduction 代表加入新 feature group 後 MAE 下降。

## V. Results

正式 artifact：

```text
artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval/
```

### A. RPE-Only 143 Sets

| Model | MAE | Spearman $\rho$ | rounded $\pm 1$ acc |
|---|---:|---:|---:|
| Baseline exercise mean | 1.3752 | 0.2704 | 0.6154 |
| Model A ridge | 1.2844 | 0.4690 | 0.6713 |
| Model A random forest | **1.2456** | **0.4770** | 0.6294 |
| Model B ridge | 2.4394 | 0.1892 | 0.3986 |
| Model B random forest | 1.4246 | 0.3278 | 0.5245 |

解讀：目前最強的是 metadata/progress random forest。直接加入 105 個 GT phase-aware IMU features 後，LOSO 泛化變差。這不表示 IMU features 沒有訊號，而是表示目前資料量下，高維 set-level features 直接進模型會過擬合，且 RPE 仍主要被 exercise、set progress、kg、n_reps 等 context/progress 驅動。

### B. RPE+VO2 Overlap 96 Sets

在 96 個同時有 RPE 與 VO2 的 sets 上，Model C 的最佳結果為：

| Lag | Model | MAE | Spearman $\rho$ | rounded $\pm 1$ acc |
|---:|---|---:|---:|---:|
| 45 s | C random forest | **1.2625** | 0.1793 | 0.6354 |
| 45 s | A random forest | 1.2849 | 0.2623 | 0.6354 |
| 45 s | B random forest | 1.2871 | 0.1595 | 0.6250 |

45 秒 VO2 lag 對 random forest 的 `C - B` 增益為：

```text
MAE reduction = 0.0247
Spearman gain = 0.0198
rounded +/-1 acc gain = 0.0104
```

解讀：VO2 在 45 秒 lag 有小幅 MAE 改善，但增益很小，且 Spearman 仍低於 Model A。這支持 005 的說法：VO2 可作 delayed physiological-load covariate，但目前不是強主訊號。

### C. Subject Calibration

Model D 使用 held-out subject 每個 exercise 的第一個 set 作 calibration offset，評估其餘 rows。Random forest 的 Spearman 在多個 lag 約落在 0.52 到 0.54，但 MAE 約 1.29 到 1.31，且評估 rows 降為 60 到 64。

解讀：few-shot subject calibration 可能改善 rank-order trend，但目前不穩定，不能當作主要結論。下一版若要主打 calibration，需固定 calibration protocol 並增加資料量。

## VI. Figure and Table Reading Guide

主要檔案：

- `metrics/nested_model_summary.csv`：所有 nested model 的 LOSO overall metrics。
- `metrics/model_delta_summary.csv`：Model B-A 與 Model C-B 的增益。
- `metrics/fold_metrics.csv`：每個 held-out subject 的 fold metrics。
- `tables/nested_model_predictions.csv`：所有模型的 out-of-subject predictions。
- `figures/rpe_only_nested_mae.png`：143-set RPE-only nested MAE。
- `figures/vo2_lag_nested_metrics.png`：每個 VO2 lag 的 MAE 與 Spearman。
- `figures/vo2_incremental_mae_gain.png`：加入 VO2 後的 MAE reduction。
- `figures/best_model_predictions.png`：最佳 feature model 的 prediction scatter。

圖表解讀：

- `rpe_only_nested_mae.png` 若 Model B 低於 Model A，代表 IMU phase features 提供增益；本次結果相反。
- `vo2_incremental_mae_gain.png` 的 y 軸大於 0 代表 VO2 對 MAE 有改善；本次只有部分 lag/model 為正，且幅度小。
- `best_model_predictions.png` 用來檢查預測是否只回歸平均值；本次點雲仍顯示 RPE 高低端預測收縮。

## VII. Limitations

- RPE+VO2 overlap 只有 4 subjects、96 sets；60 秒 lag 只有 92 rows。
- Model B 使用 105 個高維 IMU features，對 143 rows 來說特徵數偏多。
- 目前 features 是 set 完成後的 summary，不能直接視為逐 rep real-time prediction。
- 本次只使用 ground-truth CE phase features，未驗證 predicted segmentation 對特徵與 RPE 的影響。
- VO2 subject-normalized 欄位被排除在 primary model 外；若未來要使用，需在 fold 內或 online calibration protocol 中重算。
- Model D 使用 held-out subject 的 calibration labels，屬補充 few-shot 分析，不是純 subject-independent 結果。

## VIII. Reproducibility

重跑指令：

```bash
.venv311/bin/python tools/evaluate_imu_vo2_rpe_framework.py \
  --output-dir artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval
```

驗收指令：

```bash
.venv311/bin/python -m py_compile tools/evaluate_imu_vo2_rpe_framework.py
python3 -m json.tool artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval/summary.json
find artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval -maxdepth 3 -type f | sort
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```

若最後一行只命中各文件中的檢查指令本身，代表沒有新增未決標記。

## IX. Conclusion

正式驗證的主要結論是：目前資料下，RPE 跨人預測最穩的是 metadata/progress baseline；GT phase-aware IMU features 以高維形式直接加入模型會降低泛化；delayed VO2 在 45 秒 lag 對 MAE 有小幅增益，但不是強主訊號。因此，下一步不應直接做更大的模型，而應先做低維 feature selection、exercise-specific feature weighting，並設計 predicted segmentation gap 評估，確認哪些 IMU features 在部署情境下仍穩定。

## References

- `docs/tasks/005_research_framework_imu_vo2_rpe_report.md`
- `artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/summary.json`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/summary.json`
- `artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval/summary.json`
