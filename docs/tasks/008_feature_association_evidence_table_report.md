# Feature Association Evidence Table for Borg/RPE Claims

> Task ID: `008_feature_association_evidence_table`  
> Branch: `rep-segmentation-classification`  
> Task Owner: `Main Agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務把「哪些特徵和 Borg/RPE 有關聯」整理成可對外說明的 evidence table。方法分成兩層：第一層計算候選特徵與 Borg/RPE 的 Spearman $\rho$，並提供 raw、subject-centered、exercise-centered、subject+exercise-centered 四種版本；第二層引用 007 的 leave-one-subject-out ablation，檢查加入 low-dimensional IMU trend 或 delayed VO2 後是否改善模型。結果顯示，`cumulative_tut_exercise_sec` 與 `set_index_numeric` 是最強的累積暴露/流程 proxy；low-dimensional IMU trend 在 96-set VO2 overlap subset 上有 group-level 增益；45 秒 delayed VO2 尤其是 `vo2_slope` 有輔助證據。所有結論限定為 association evidence，不宣稱因果。

## Index Terms

IMU, Borg RPE, VO2, feature association, Spearman correlation, leave-one-subject-out, cumulative TUT

## I. Introduction

前一輪 007 驗證回答了「low-dimensional IMU trend + delayed VO2 是否能改善 RPE 預測」。但若要向其他人證明題目的合理性，還需要更直接的表格說明：哪些特徵和 RPE 有關、這種關聯是否只是 subject 或 exercise 差異造成、以及加入這些特徵後模型是否真的有增益。

本任務產出一份可引用的 feature association evidence table，讓論文或簡報可以用保守句型描述：

```text
這些特徵和 Borg/RPE 有可觀察的關聯與控制後預測增益，
但目前證據仍是 association，不是 causal fatigue mechanism。
```

## II. Task Definition

- Goal: 建立每個候選特徵與 Borg/RPE 關聯的可追溯 evidence table。
- Inputs:
  - `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
  - `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`
  - `artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval/metrics/model_delta_summary.csv`
- Allowed Changes:
  - 新增 `tools/build_feature_association_evidence_table.py`
  - 新增 `artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table/`
  - 更新 `artifacts/fatigue_rpe_vo2/RESULTS_INDEX.md`
  - 更新 `docs/README.md`、`todo.md`
- Forbidden Changes:
  - 不改資料 schema
  - 不改 007 模型指標
  - 不刪除或回復任何既有 artifact
  - 不處理 `datasets/raw_data` dirty deletion 狀態
- Non-goals:
  - 不做因果推論
  - 不新增大型模型
  - 不宣稱 lowdim IMU trend 是即時逐 rep predictor
- Outputs:
  - feature evidence table、group model evidence table、correlation long table、figures、summary JSON、task report
- Acceptance Criteria:
  - Spearman correlation 至少包含 raw、subject-centered、exercise-centered
  - model evidence 必須引用 007 的 LOSO ablation
  - 明確標示 `set_index_numeric` 是 diagnostic proxy
  - 明確區分單 set TUT 與 cumulative TUT
- Dependencies:
  - 007 lowdim set trend + delayed VO2 validation
- Handoff:
  - 下一步可做 cumulative TUT vs set order 的獨立 ablation，以及 real-time prefix features

## III. Input Data and Assumptions

使用 021 的 143 sets、6 subjects 估計 workload、cumulative dose、set-order 與 IMU trend 特徵的 RPE association。使用 022 中 `lag_sec = 45` 的 96 sets、4 subjects 估計 delayed VO2 特徵，因為 007 中 45 秒 lag 是 VO2 增益最明顯的條件。

本任務額外衍生一個分析用特徵：

```text
cumulative_tut_exercise_sec
```

定義為同一 `folder + exercise` 內，按照 `set_index_numeric` 排序後，截至目前 set 的 `total_tut_sec` 累積和。此欄位只存在於 008 evidence artifact，不改原始 CSV schema。

## IV. Method

### A. Feature-Level Association

對每個候選特徵 $x$ 與 Borg/RPE $y$ 計算 Spearman rank correlation：

```text
$\rho_s = corr(rank(x), rank(y))$
```

為了檢查 subject 與 exercise confounding，另計算 centered 版本：

```text
$\tilde{x}_i = x_i - \bar{x}_{g(i)}$
```

其中 $g(i)$ 分別為 subject、exercise 或 subject+exercise group。p-value 另外用 Benjamini-Hochberg FDR 產生 q-value，作為多個候選特徵的探索性校正。

### B. Group-Level Controlled Evidence

模型增益引用 007 的 leave-one-subject-out ablation：

```text
$\Delta MAE = MAE_{baseline} - MAE_{augmented}$
```

正值代表加入該特徵組後錯誤下降。主要引用三個 group-level evidence：

- `order_diagnostic_gain`
- `B_minus_A_lowdim_imu_gain`
- `C_minus_B_vo2_gain`

## V. Results

正式 artifact：

```text
artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table/
```

### A. Feature-Level Evidence

| Feature | Family | Raw $\rho$ | Subject-centered $\rho$ | Exercise-centered $\rho$ | Subject+Exercise-centered $\rho$ | Claim |
|---|---|---:|---:|---:|---:|---|
| `cumulative_tut_exercise_sec` | cumulative dose | 0.4987 | 0.5395 | 0.5889 | 0.7791 | feature-level association |
| `set_index_numeric` | diagnostic proxy | 0.4397 | 0.4982 | 0.5017 | 0.7997 | strong diagnostic proxy only |
| `vo2_slope` | delayed VO2 45 s | 0.3639 | 0.2707 | 0.2193 | 0.0475 | delayed physiological evidence |
| `pca_diff_rms_mean` | lowdim IMU trend | -0.2861 | -0.2776 | -0.2602 | -0.1921 | group-level IMU evidence |
| `gyro_diff_gain_last2_vs_first2` | lowdim IMU trend | 0.1958 | 0.3228 | 0.1146 | 0.1319 | group-level IMU evidence |
| `gyro_mag_diff_rms_slope` | lowdim IMU trend | 0.1928 | 0.2464 | 0.0567 | 0.0956 | group-level IMU evidence |
| `total_tut_sec` | workload dose | -0.0033 | -0.0452 | 0.0163 | -0.2734 | control only |

重點解讀：

1. `cumulative_tut_exercise_sec` 比單一 set 的 `total_tut_sec` 更符合疲勞累積敘事。
2. `set_index_numeric` 很強，但它是實驗順序與累積暴露 proxy，不能被包裝成 IMU 特徵。
3. IMU 單特徵強度中等，較適合用 feature group 的模型增益來支持。
4. VO2 單特徵中，45 秒 `vo2_slope` 最可講，但 subject+exercise-centered 後下降，表示它仍受流程與個體差異影響。

### B. Controlled Model Evidence

| Feature Group | Dataset | Lag | Comparison | MAE reduction | Spearman gain | $\pm1$ acc gain |
|---|---|---:|---|---:|---:|---:|
| set order diagnostic | 143-set RPE | NA | order diagnostic gain | 0.2878 | 0.3889 | 0.1259 |
| lowdim IMU trend | 96-set VO2 overlap | 45 s | B - A | 0.2271 | 0.3484 | 0.1042 |
| lowdim IMU trend | 143-set RPE | NA | B - A | 0.0622 | 0.0525 | 0.0559 |
| delayed VO2 | 96-set VO2 overlap | 45 s | C - B | 0.0906 | 0.1087 | 0.0521 |

最保守、可對外引用的說法是：

```text
在目前資料中，累積 TUT / set order proxy 與 RPE 的關聯最強；
low-dimensional IMU trend 在控制 workload/dose 後仍提供 group-level 預測增益；
45 秒 delayed VO2 對 lowdim IMU model 有額外但較小的輔助增益。
```

## VI. Figure and Table Reading Guide

主要檔案：

- `tables/feature_association_evidence.csv`：每個候選特徵的主要 evidence table。
- `tables/group_model_evidence.csv`：007 LOSO ablation 的 group-level model evidence。
- `metrics/feature_correlation_long.csv`：long format correlation table，包含 raw 與 centered variants。
- `figures/feature_spearman_evidence.png`：raw、subject-centered、exercise-centered $\rho$ 橫條圖。
- `figures/group_model_gain_evidence.png`：feature group 的 MAE reduction。

讀圖方式：

- `feature_spearman_evidence.png`：bar 越遠離 0 代表 monotonic association 越強；正負號代表方向，不代表好壞。
- `group_model_gain_evidence.png`：MAE reduction 大於 0 代表加入該 group 後 LOSO error 下降。
- `feature_association_evidence.csv` 中的 `claim_strength` 已把 proxy、control、IMU group evidence、delayed physiological evidence 分開，避免過度宣稱。

## VII. Limitations

- Spearman correlation 不是因果證明。
- `cumulative_tut_exercise_sec` 和 `set_index_numeric` 高度接近實驗流程與累積暴露，需在下一輪 ablation 中分開測試。
- VO2 結論只來自 45 秒 lag、96 sets、4 subjects。
- Lowdim IMU trend features 目前仍是 post-set summary，不是 rep-by-rep real-time features。
- 單一特徵的方向可能受 exercise programming 影響；例如 `n_reps` 與 RPE 的負相關不應直接解讀為「次數越多越不累」。

## VIII. Reproducibility

重跑指令：

```bash
.venv311/bin/python tools/build_feature_association_evidence_table.py \
  --output-dir artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table
```

驗收指令：

```bash
.venv311/bin/python -m py_compile tools/build_feature_association_evidence_table.py
python3 -m json.tool artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table/summary.json
find artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table -maxdepth 3 -type f | sort
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```

## IX. Conclusion

本任務完成一份可對外引用的 feature association evidence table。現階段最穩的研究敘事不是「某個單一 IMU 特徵直接代表疲勞」，而是「累積暴露最強、lowdim IMU trend 有 group-level 增益、45 秒 delayed VO2 有額外輔助增益」。下一步應把 `cumulative_tut_exercise_sec`、`set_index_numeric` 與 lowdim IMU trend 放進同一個 ablation，測試累積 TUT 是否能取代單純 set order proxy。

## References

- `artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval/metrics/model_delta_summary.csv`
- `artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table/summary.json`
- `docs/tasks/007_lowdim_set_trend_vo2_validation_report.md`
