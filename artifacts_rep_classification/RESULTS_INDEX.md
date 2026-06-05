# Rep / Phase / Fatigue Results Index

最後更新：2026-05-17

本 root 已依「論文敘事鏈」瘦身，只保留能支撐目前專題論文敘事的核心結果。被刪除的早期 artifacts 屬於已被取代版本、smoke test、舊未編號輸出或大量可重生 waveform 圖。

後續新實驗請改輸出到分類 root：

```text
artifacts/<domain>/<experiment_id>_<short_slug>/
```

詳細規範見 `docs/artifact_organization_zh.md` 與 `docs/tasks/003_artifact_taxonomy_report.md`。

## 保留集

```text
014_literature_inspired_rep_methods/
015_boundary_candidate_recall_analysis/
016_dense_candidate_dp_decoder/
017_phase_split_dcp_dp_fs/
018_borg_gt_waveform_relation/
019_vo2_gt_waveform_relation/
021_rpe_feature_correlation_with_yushuan/
022_realtime_rpe_vo2_feature_correlation/
023_phase_aware_fatigue_ce_rpe_analysis/
024_imu_fatigue_component_relevance_figure/
```

## 024_imu_fatigue_component_relevance_figure

目的：產生論文 / 簡報用總結圖，說明 IMU / VO2 成分與 Borg/RPE 的關聯。

重點：

- Accumulated TUT：raw Spearman `0.4594`
- Delayed VO2 slope：raw Spearman `0.3639`
- VO2 baseline delta：raw Spearman `-0.3500`
- CE phase range：raw Spearman `0.3377`
- CE phase similarity：raw Spearman `-0.3272`

重點檔案：

- `024_imu_fatigue_component_relevance_figure/summary.json`
- `024_imu_fatigue_component_relevance_figure/024_imu_fatigue_component_relevance_summary.png`
- `024_imu_fatigue_component_relevance_figure/024_imu_fatigue_component_relevance_table.csv`
- `docs/imu_fatigue_component_relevance_024_paper_zh.md`

解讀：IMU 不能直接量測肌肉疲勞，但可量化疲勞相關動作學變化；最穩定的是累積 TUT、CE phase range、phase similarity drift 與 VO2 延遲負荷。

## 023_phase_aware_fatigue_ce_rpe_analysis

目的：用 ground-truth concentric / eccentric phase 檢查 phase-aware features 與 Borg/RPE 的關係。

重點：

- rep rows：`1677`
- set rows：`143`
- cumulative active time raw Spearman 約 `0.4594`
- eccentric PCA range mean raw Spearman 約 `0.3377`
- eccentric waveform similarity drift raw Spearman 約 `-0.3272`

重點檔案：

- `023_phase_aware_fatigue_ce_rpe_analysis/summary.json`
- `023_phase_aware_fatigue_ce_rpe_analysis/023_phase_aware_set_correlations.csv`
- `023_phase_aware_fatigue_ce_rpe_analysis/023_phase_aware_set_correlations_by_exercise.csv`
- `docs/phase_aware_fatigue_ce_rpe_023_zh.md`

解讀：數據支持 CE phase-aware fatigue 方向，但不支持只靠「向心速度下降」作為單一規則。

## 022_realtime_rpe_vo2_feature_correlation

目的：合併 RPE set-level features 與 VO2 lag window，分析即時 RPE 估計的 IMU/VO2 特徵。

來源依賴：

- `019_vo2_gt_waveform_relation/019_vo2_set_waveform_dataset.csv`
- `021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`

重點：

- rows：`572`
- sets：`96`
- lags：`0, 10, 20, 30, 45, 60` 秒

重點檔案：

- `022_realtime_rpe_vo2_feature_correlation/summary.json`
- `022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_merged_set_dataset.csv`
- `022_realtime_rpe_vo2_feature_correlation/022_realtime_rpe_vo2_feature_correlations.csv`
- `docs/realtime_rpe_vo2_features_022_zh.md`

解讀：即時 RPE 主訊號仍來自 IMU set-level fatigue state；VO2 有訊號，但受延遲、休息與個人基準影響，適合作輔助。

## 021_rpe_feature_correlation_with_yushuan

目的：保留 022 的 RPE set-level 來源資料。

瘦身後只保留：

- `021_rpe_feature_correlation_with_yushuan/summary.json`
- `021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
- `docs/rpe_feature_correlation_020_zh.md`

解讀：完整圖表已瘦身；若需重跑 022，保留的 set-level feature dataset 是主要串接輸入。

## 019_vo2_gt_waveform_relation

目的：保留 022 的 VO2 set-level 來源資料。

瘦身後只保留：

- `019_vo2_gt_waveform_relation/summary.json`
- `019_vo2_gt_waveform_relation/019_vo2_set_waveform_dataset.csv`

解讀：完整 VO2 診斷圖已瘦身；若需重跑 022，保留的 VO2 set waveform dataset 是主要串接輸入。

## 018_borg_gt_waveform_relation

目的：使用 ground-truth rep / phase segmentation 測試 waveform/TUT features 是否含有 Borg/RPE 訊號。

重點：

- merged ground-truth reps：`1677`
- trainable subjects：`6`
- 測試的是自動切割雜訊前的特徵上限。

重點檔案：

- `018_borg_gt_waveform_relation/summary.json`
- `018_borg_gt_waveform_relation/018_gt_rep_waveform_borg_dataset.csv`
- `018_borg_gt_waveform_relation/018_borg_prediction_summary.csv`
- `docs/borg_waveform_relation_018_zh.md`

## 017_phase_split_dcp_dp_fs

目的：使用 016 的 `DCP-DP-FS` reps 檢查 concentric / eccentric phase split。

重點：

- true phase segments：`5364`
- predicted phase segments：`5308`
- method：`pca-reversal`

重點檔案：

- `017_phase_split_dcp_dp_fs/summary.json`
- `017_phase_split_dcp_dp_fs/phase_split_metrics.csv`
- `017_phase_split_dcp_dp_fs/phase_tut_error_summary.csv`
- `docs/phase_split_diagnostics_017_zh.md`

解讀：全量 `phase_waveforms/` 已瘦身刪除；若需診斷圖，使用工具重新產生。

## 016_dense_candidate_dp_decoder

目的：用 dense candidate pool + dynamic programming 改善高 IoU rep boundary。

重點：

- active blocks：`239`
- true reps：`2720`
- methods：`DCP-DP`, `DCP-DP-FS`

重點檔案：

- `016_dense_candidate_dp_decoder/summary.json`
- `016_dense_candidate_dp_decoder/016_dense_candidate_dp_comparison.csv`
- `016_dense_candidate_dp_decoder/016_dense_candidate_dp_comparison_table.png`
- `docs/dense_candidate_dp_decoder_016_zh.md`

解讀：全量 `waveform_all_sets/` 已瘦身刪除；保留方法比較、by-exercise/by-subject 與主圖。

## 015_boundary_candidate_recall_analysis

目的：保留 016 方法動機的最小證據，說明瓶頸在 candidate selection / decoding。

重點：

- true internal boundaries：`2481`
- `raw_gyro_energy_valleys` 在 20 samples 內 recall `0.9964`
- `fusion_candidate_pool` 在 20 samples 內 recall `0.8130`

瘦身後保留：

- `015_boundary_candidate_recall_analysis/summary.json`
- `015_boundary_candidate_recall_analysis/015_boundary_candidate_recall_summary.csv`
- `015_boundary_candidate_recall_analysis/015_boundary_candidate_recall_by_exercise.csv`
- `015_boundary_candidate_recall_analysis/015_candidate_recall_by_source.png`
- `docs/boundary_candidate_recall_015_zh.md`

## 014_literature_inspired_rep_methods

目的：比較文獻啟發 baseline 與 `LIFT-Fusion`，建立 016 前的 rep segmentation 參考點。

重點：

- active blocks：`239`
- true reps：`2720`
- methods：`STAYFIT-BA`, `MAXXYT-MAP`, `MFIT-FSTE`, `CARA-DTW-FS`, `LIFT-Fusion`

重點檔案：

- `014_literature_inspired_rep_methods/summary.json`
- `014_literature_inspired_rep_methods/014_literature_method_comparison.csv`
- `014_literature_inspired_rep_methods/014_literature_method_comparison_table.png`
- `docs/literature_inspired_rep_methods_014_zh.md`

解讀：全量 `waveform_all_sets/` 已瘦身刪除；保留比較表與主圖。

## 清理紀錄

2026-05-17 清理：

- 刪除早期 001-013 非保留 artifacts。
- 刪除 012 smoke/formal sequence model artifacts。
- 刪除 018 exclude_sparse、020 舊 RPE 分析。
- 刪除舊未編號 results：`labels_*`、`dominant_axis_*`、`pca_*`、`short_time_energy_*`、`methods_comparison`、`waveform_method_comparison`。
- 刪除保留集內大量可重生圖：014/016 `waveform_all_sets/`、017 `phase_waveforms/`。

詳細決策見 `docs/tasks/002_artifact_cleanup_report.md`。
