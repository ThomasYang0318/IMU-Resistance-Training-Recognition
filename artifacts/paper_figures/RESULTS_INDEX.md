# Paper Figures Results Index

最後更新：2026-05-17

本 root 保存可放入論文、簡報或研究設計文件的圖表產物。此 root 不存放大型 raw predictions，也不取代正式模型實驗 root。

## 001_imu_vo2_rpe_framework

目的：設計 IMU、VO2 ground truth 與 Borg/RPE 關聯的整體研究框架，並產出 paper-ready 框架圖與引用對應表。

輸入：

- `docs/borg_waveform_relation_018_zh.md`
- `docs/realtime_rpe_vo2_features_022_zh.md`
- `docs/phase_aware_fatigue_ce_rpe_023_zh.md`
- `docs/imu_fatigue_component_relevance_024_paper_zh.md`
- `artifacts_rep_classification/018_borg_gt_waveform_relation/`
- `artifacts_rep_classification/019_vo2_gt_waveform_relation/`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/`
- `artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/`
- `artifacts_rep_classification/024_imu_fatigue_component_relevance_figure/`

方法摘要：以既有 GT rep/CE phase 分析為上限證據，整合 RPE/Borg、VO2 lag 與 IMU feature relevance 文獻，形成 phase-aware、exercise-aware、subject-calibrated 的研究設計。

Key files：

- `001_imu_vo2_rpe_framework/summary.json`
- `001_imu_vo2_rpe_framework/figures/imu_vo2_rpe_research_framework.png`
- `001_imu_vo2_rpe_framework/tables/citation_traceability.csv`
- `docs/tasks/005_research_framework_imu_vo2_rpe_report.md`

結論：IMU 應被定位為 fatigue-related movement changes 的量化工具；VO2 是 delayed physiological-load covariate；RPE/Borg 是 subjective exertion label；後續模型需使用 subject-wise validation、exercise-aware phase features 與 subject calibration。
