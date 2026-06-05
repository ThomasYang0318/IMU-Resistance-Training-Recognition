# Fatigue / RPE / VO2 Results Index

最後更新：2026-05-17

本 root 保存 fatigue、RPE/Borg 與 VO2 融合相關的正式新實驗。歷史保留結果仍在 `artifacts_rep_classification/`，新 formal experiment 從此 root 遞增。

## 001_gt_phase_imu_vo2_rpe_framework_eval

目的：正式驗證 005 研究框架，測試 GT phase-aware IMU features 與 delayed VO2 是否能在 leave-one-subject-out 條件下改善 Borg/RPE 預測。

輸入：

- `artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/023_phase_aware_set_feature_dataset.csv`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`

方法摘要：

- RPE-only：143 sets、6 subjects。
- RPE+VO2 overlap：96 sets、4 subjects，逐 lag 評估 `0, 10, 20, 30, 45, 60` 秒。
- Split：leave-one-subject-out。
- Models：baseline exercise mean、metadata/progress、metadata+IMU phase、metadata+IMU phase+delayed VO2、few-shot subject calibration。

Primary metrics：

| Setting | Best / Key Model | MAE | Spearman | rounded +/-1 acc |
|---|---|---:|---:|---:|
| RPE-only 143 sets | Model A random forest | 1.2456 | 0.4770 | 0.6294 |
| RPE-only 143 sets | Model B random forest | 1.4246 | 0.3278 | 0.5245 |
| RPE+VO2 96 sets | Model C random forest, lag 45 s | 1.2625 | 0.1793 | 0.6354 |

結論：

- Metadata/progress baseline 目前比高維 IMU phase features 更穩。
- 直接加入 105 個 GT phase-aware IMU features 造成跨人泛化下降。
- VO2 在 45 秒 lag 對 random forest 有小幅 MAE 改善，但增益很小且不穩定。
- 下一步應做低維 feature selection、exercise-specific weighting 與 predicted segmentation gap 評估。

Key files：

- `001_gt_phase_imu_vo2_rpe_framework_eval/summary.json`
- `001_gt_phase_imu_vo2_rpe_framework_eval/metrics/nested_model_summary.csv`
- `001_gt_phase_imu_vo2_rpe_framework_eval/metrics/model_delta_summary.csv`
- `001_gt_phase_imu_vo2_rpe_framework_eval/tables/nested_model_predictions.csv`
- `001_gt_phase_imu_vo2_rpe_framework_eval/figures/rpe_only_nested_mae.png`
- `001_gt_phase_imu_vo2_rpe_framework_eval/figures/vo2_lag_nested_metrics.png`
- `docs/tasks/006_gt_phase_imu_vo2_rpe_formal_validation_report.md`

## 002_lowdim_set_trend_vo2_eval

目的：驗證不用 CE phase-specific features 時，低維 set-level IMU trend 與 delayed VO2 是否能改善 Borg/RPE 預測。

輸入：

- `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`

方法摘要：

- RPE-only：143 sets、6 subjects。
- RPE+VO2 overlap：96 sets、4 subjects，逐 lag 評估 `0, 10, 20, 30, 45, 60` 秒。
- 主模型不使用 `concentric_*`、`eccentric_*`、CE ratio 或 subject-normalized VO2。
- `set_index_numeric` 只作 diagnostic，不作純 IMU/生理特徵。

Primary metrics：

| Setting | Best / Key Model | MAE | Spearman | rounded +/-1 acc |
|---|---|---:|---:|---:|
| RPE-only 143 sets | A + set order diagnostic RF | 1.3595 | 0.3964 | 0.6154 |
| RPE-only 143 sets | B lowdim set trend RF | 1.5852 | 0.0600 | 0.5455 |
| RPE+VO2 96 sets | C lowdim + VO2 RF, lag 45 s | 1.2300 | 0.2797 | 0.6875 |
| RPE+VO2 92 sets | B lowdim ridge, lag 60 s | 1.2786 | 0.3790 | 0.6413 |

結論：

- 低維 non-phase features 比 105 個 phase features 更適合作第一版方向。
- 在 96-set overlap subset 上，lowdim IMU trend 相對 workload/TUT baseline 有明顯增益。
- Delayed VO2 在 45 秒 lag 對 random forest 有最佳 MAE 增益。
- 在 143-set full RPE-only 資料上，set order / exercise context 仍比 lowdim IMU trend 更穩。

Key files：

- `002_lowdim_set_trend_vo2_eval/summary.json`
- `002_lowdim_set_trend_vo2_eval/metrics/nested_model_summary.csv`
- `002_lowdim_set_trend_vo2_eval/metrics/model_delta_summary.csv`
- `002_lowdim_set_trend_vo2_eval/tables/nested_model_predictions.csv`
- `002_lowdim_set_trend_vo2_eval/figures/rpe_lowdim_nested_mae.png`
- `002_lowdim_set_trend_vo2_eval/figures/vo2_lowdim_lag_metrics.png`
- `docs/tasks/007_lowdim_set_trend_vo2_validation_report.md`

## 003_feature_association_evidence_table

目的：把「哪些特徵和 Borg/RPE 有關聯」整理成可引用的 evidence table，區分單特徵 Spearman association、subject/exercise-centered association，以及 007 LOSO ablation 的 group-level model evidence。

輸入：

- `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`
- `002_lowdim_set_trend_vo2_eval/metrics/model_delta_summary.csv`

方法摘要：

- 021 RPE-only：143 sets、6 subjects。
- 022 VO2 lag 45 s：96 sets、4 subjects。
- Spearman association：raw、subject-centered、exercise-centered、subject+exercise-centered。
- 額外衍生 `cumulative_tut_exercise_sec`，代表同 subject + exercise 內截至目前 set 的累積 TUT。

Primary evidence：

| Feature / Group | Evidence | Value |
|---|---|---:|
| `cumulative_tut_exercise_sec` | raw Spearman with RPE | 0.4987 |
| `set_index_numeric` | raw Spearman with RPE | 0.4397 |
| `vo2_slope`, lag 45 s | raw Spearman with RPE | 0.3639 |
| lowdim IMU trend group | B-A MAE reduction, 96-set overlap | 0.2271 |
| delayed VO2 group, lag 45 s | C-B MAE reduction, 96-set overlap | 0.0906 |

結論：

- 累積 TUT 與 set order proxy 是目前和 RPE 最強的累積暴露證據。
- Lowdim IMU trend 的單特徵相關中等，但在 96-set overlap subset 上有 group-level 預測增益。
- Delayed VO2 的 45 秒 slope 有單特徵 association，且 VO2 group 對 lowdim model 有額外 MAE 改善。
- 所有敘事都應限定為 association evidence，不宣稱因果。

Key files：

- `003_feature_association_evidence_table/summary.json`
- `003_feature_association_evidence_table/tables/feature_association_evidence.csv`
- `003_feature_association_evidence_table/tables/group_model_evidence.csv`
- `003_feature_association_evidence_table/metrics/feature_correlation_long.csv`
- `003_feature_association_evidence_table/figures/feature_spearman_evidence.png`
- `003_feature_association_evidence_table/figures/group_model_gain_evidence.png`
- `docs/tasks/008_feature_association_evidence_table_report.md`

## 004_controlled_one_feature_ablation

目的：回應「每個 exercise 的 RPE 都從 1 開始」造成的 progression confounding，使用 controlled one-feature-at-a-time ablation，檢查單一 workload、IMU trend 或 delayed VO2 feature 是否能在控制 exercise + progression 後仍有增益。

輸入：

- `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`

方法摘要：

- Model：Ridge regression。
- Split：leave-one-subject-out。
- Baselines：
  - M0：exercise mean。
  - M1：exercise + set index。
  - M2：exercise + cumulative TUT。
  - M3：exercise + set index + cumulative TUT。
- Candidate：M3 每次只新增一個 numeric feature。
- 嚴格判讀：candidate 必須超過最佳 progression baseline 才算強證據。

Primary evidence：

| Dataset | Best Progression Baseline | MAE | Spearman | Result |
|---|---|---:|---:|---|
| RPE-only 143 sets | exercise + cumulative TUT | 1.2206 | 0.5771 | no single added feature beats this |
| VO2 lag45 96 sets | exercise + cumulative TUT | 1.0579 | 0.5381 | no single added feature beats this |

結論：

- 目前資料最強證據是 within-exercise cumulative progression。
- 單一 IMU/VO2 feature 對 M3 有時有小幅改善，但沒有任何一個特徵超過最佳 progression baseline。
- 不能宣稱單一 IMU 或 delayed VO2 特徵獨立證明疲勞；下一步應做 exercise-specific interaction 或 feature group ablation。

Key files：

- `004_controlled_one_feature_ablation/summary.json`
- `004_controlled_one_feature_ablation/tables/controlled_one_feature_ablation.csv`
- `004_controlled_one_feature_ablation/metrics/model_summary.csv`
- `004_controlled_one_feature_ablation/metrics/fold_metrics.csv`
- `004_controlled_one_feature_ablation/figures/progression_baselines_mae.png`
- `004_controlled_one_feature_ablation/figures/one_feature_delta_vs_best_progression.png`
- `docs/tasks/009_controlled_one_feature_ablation_report.md`

## 005_module_necessity_rpe_vo2

目的：同時回答「哪個單一 feature 最相關」與「為什麼系統需要動作切分、set/rep/TUT 特徵與 VO2」。分析採開放式 single-feature ranking，不預設動作切分或 VO2 一定重要。

輸入：

- `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`
- `artifacts_rep_classification/019_vo2_gt_waveform_relation/summary.json`

方法摘要：

- Single-feature ranking：所有 numeric features 的 raw / centered Spearman 與單特徵 LOSO Ridge MAE。
- Exercise context：作為 categorical context，用 exercise mean 對 global mean 的 MAE reduction 評估。
- Module ladder：global mean -> exercise context -> set index -> cumulative TUT -> lowdim IMU group -> delayed VO2。
- VO2 分成兩個命題：對 RPE 的增益、以及 VO2 作為獨立生理 target 是否可由 IMU 估計。

Primary evidence：

| Question | Best / Key Evidence | Result |
|---|---|---:|
| 最相關單一 numeric feature | `cumulative_tut_exercise_sec`, RPE-only raw Spearman | 0.4987 |
| 最相關單一 numeric feature, VO2 overlap | `cumulative_tut_exercise_sec` raw Spearman | 0.5760 |
| 動作切分是否有用 | exercise mean vs global mean, RPE-only MAE reduction | 0.1754 |
| TUT segmentation 是否有用 | cumulative TUT vs set index, VO2 overlap MAE reduction | 0.2472 |
| VO2 對 RPE 是否必要 | delayed VO2 after IMU, VO2 overlap MAE reduction | -0.1255 |
| VO2 是否可估計 | 019 VO2 mean lag10 RF Spearman | 0.6798 |

結論：

- 最強單一 RPE 特徵是 cumulative TUT，不是 VO2。
- 動作切分/辨識有證據，因為 exercise context 明顯優於 global mean，且 set progression 必須綁定 exercise 才合理。
- set/rep/TUT segmentation 有證據，因為 cumulative TUT 是最強單一特徵。
- 目前不支持 VO2 是 RPE 預測必要輸入；VO2 更適合作為延遲生理負荷的獨立估計 target。

Key files：

- `005_module_necessity_rpe_vo2/summary.json`
- `005_module_necessity_rpe_vo2/tables/single_feature_open_ranking.csv`
- `005_module_necessity_rpe_vo2/tables/module_necessity_comparisons.csv`
- `005_module_necessity_rpe_vo2/tables/vo2_estimability_evidence.csv`
- `005_module_necessity_rpe_vo2/metrics/rpe_module_ladder.csv`
- `005_module_necessity_rpe_vo2/figures/single_feature_raw_spearman_ranking.png`
- `005_module_necessity_rpe_vo2/figures/rpe_module_ladder_mae.png`
- `005_module_necessity_rpe_vo2/figures/vo2_estimability_spearman.png`
- `docs/tasks/010_module_necessity_single_feature_ranking_report.md`
