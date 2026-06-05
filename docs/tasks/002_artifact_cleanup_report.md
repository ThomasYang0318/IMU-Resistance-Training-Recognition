# Artifact Cleanup Report

> Task ID: `002`  
> Branch: `agent/002-artifact-cleanup`  
> Task Owner: `artifact-cleanup sub-agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務依照「論文敘事鏈」積極瘦身 artifacts。保留 active detection baseline 與 rep segmentation、phase split、Borg/RPE/VO2 fatigue 相關核心結果；同時保留 015、019、021 的最小可追溯證據，避免切斷 016 與 022 的來源脈絡。刪除早期被取代版本、smoke tests、舊未編號 artifacts 與大量可重生 waveform/phase 圖。

## Index Terms

artifacts, cleanup, IMU, repetition segmentation, phase segmentation, fatigue, RPE

## I. Introduction

現有 artifacts 包含 001-024 多輪實驗與大量診斷圖。為降低 repo 體積與閱讀成本，本任務保留論文敘事鏈必要結果，將可重生或已被取代的輸出直接刪除。

## II. Task Definition

- Goal：直接刪除非保留 artifacts，保留論文敘事鏈。
- Inputs：`artifacts_active_detection/`、`artifacts_rep_classification/`、`RESULTS_INDEX.md`。
- Allowed Changes：刪除非保留 artifact 目錄與大量可重生圖子目錄，更新結果索引。
- Forbidden Changes：不刪原始資料、不刪 tools、不刪 docs 論文敘事文件。
- Outputs：瘦身後 artifact root、更新後 results index。
- Acceptance Criteria：保留集存在且有 `summary.json`。
- Dependencies：使用者已確認「直接刪除」與「論文敘事鏈」。

## III. Input Data and Assumptions

保留集：

- `artifacts_active_detection/001_window_rf_action_5fold`
- `artifacts_rep_classification/014_literature_inspired_rep_methods`
- `artifacts_rep_classification/015_boundary_candidate_recall_analysis` 的 `summary.json`、核心 CSV 與代表圖
- `artifacts_rep_classification/016_dense_candidate_dp_decoder`
- `artifacts_rep_classification/017_phase_split_dcp_dp_fs`
- `artifacts_rep_classification/018_borg_gt_waveform_relation`
- `artifacts_rep_classification/019_vo2_gt_waveform_relation` 的 `summary.json` 與 `019_vo2_set_waveform_dataset.csv`
- `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan` 的 `summary.json` 與 `020_rpe_set_level_feature_dataset.csv`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation`
- `artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis`
- `artifacts_rep_classification/024_imu_fatigue_component_relevance_figure`

## IV. Method

刪除策略分成兩類：

1. 整個 artifact 目錄刪除：早期被取代版本、smoke tests、舊未編號結果。
2. 保留 artifact 內瘦身：刪除 `waveform_all_sets/`、`phase_waveforms/` 等可由工具重生的大量圖。

## V. Results

保留核心 CSV、PNG、`summary.json` 與 index。被刪除的舊 artifacts 可由原本工具與 `RESULTS_INDEX.md` 記錄重新產生。019/021 只保留支撐 022 追溯的 set-level 來源資料。

## VI. Figure and Table Reading Guide

保留圖表閱讀方式：

- Method comparison bar/table：比較不同 rep segmentation 方法的 IoU/F1；較高代表切割較準。
- Phase split IoU figure：比較 predicted phase 與 ground truth phase 的 overlap；IoU 越高越好。
- Fatigue/RPE correlation figure：Spearman $\rho$ 表示 monotonic association；正負號代表趨勢方向，不代表因果。
- Paper component figure：用於論文摘要 IMU/VO2 與 RPE 的關聯，應搭配限制說明閱讀。

## VII. Limitations

直接刪除會讓舊圖無法離線瀏覽；若後續需要，只能依工具重跑或從 Git 恢復。清理不改變任何演算法結果。

## VIII. Reproducibility

```bash
find artifacts_active_detection artifacts_rep_classification -mindepth 1 -maxdepth 1 -type d | sort
for d in artifacts_active_detection/001_window_rf_action_5fold artifacts_rep_classification/014_literature_inspired_rep_methods artifacts_rep_classification/015_boundary_candidate_recall_analysis artifacts_rep_classification/016_dense_candidate_dp_decoder artifacts_rep_classification/017_phase_split_dcp_dp_fs artifacts_rep_classification/018_borg_gt_waveform_relation artifacts_rep_classification/019_vo2_gt_waveform_relation artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis artifacts_rep_classification/024_imu_fatigue_component_relevance_figure; do test -f "$d/summary.json" && echo "OK $d"; done
```

## IX. Conclusion

Artifacts 清理以論文敘事鏈為核心，保留可支撐簡報與論文的結果，同時移除大量可重生或已被取代的產物。

## References

- `artifacts_active_detection/RESULTS_INDEX.md`
- `artifacts_rep_classification/RESULTS_INDEX.md`
