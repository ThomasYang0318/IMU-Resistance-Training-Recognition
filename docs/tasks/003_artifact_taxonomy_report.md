# Artifact Taxonomy Report

> Task ID: `003`  
> Branch: `agent/003-artifact-taxonomy`  
> Task Owner: `artifact-taxonomy sub-agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務規劃後續正式實驗的新 artifacts 分類 root。未來新產物不再混入舊 `artifacts_rep_classification/`，而是依 active set、rep segmentation、phase segmentation、recognition、features、fatigue、deployment 與 paper figures 分類。

## Index Terms

artifact taxonomy, experiment output, summary schema, reproducibility

## I. Introduction

舊 artifact root 名稱反映早期 rep classification 方向，已不足以涵蓋 phase split、quality/fatigue、RPE/VO2 與 deployment。新分類 root 讓每次實驗能依性質歸檔。

## II. Task Definition

- Goal：建立後續新 artifacts 的分類 root 與最小 schema。
- Inputs：現有 artifacts root、`docs/artifact_organization_zh.md`。
- Allowed Changes：新增空分類目錄與 `.gitkeep`，更新規範文件。
- Forbidden Changes：不搬移既有保留 artifacts。
- Outputs：`artifacts/` 分類 root 與規範摘要。
- Acceptance Criteria：新 root 存在，未來實驗能按 domain 放置。
- Dependencies：artifact cleanup 完成後仍保留舊 root 作歷史結果。

## III. Input Data and Assumptions

新 root 用於未來正式實驗；舊 root 只保留歷史論文敘事鏈。`scratch/` 不進正式 index。

## IV. Method

建立 domain：

```text
artifacts/
  active_set/
  rep_segmentation/
  phase_segmentation/
  recognition/
  features_quality/
  fatigue_rpe_vo2/
  deployment/
  paper_figures/
  scratch/
```

正式實驗結構：

```text
artifacts/<domain>/<experiment_id>_<short_slug>/
  summary.json
  run_config.yaml
  manifest.csv
  metrics/
  tables/
  figures/
  diagnostics/
  logs/
```

## V. Results

新分類 root 已可作為後續實驗輸出位置。最小 `summary.json` 欄位包括 `schema_version`、`experiment_id`、`name`、`domain`、`created_at`、`status`、`task`、`question`、`input_data`、`input_artifacts`、`output_dir`、`command`、`git_commit`、`split`、`primary_metrics`、`key_files`、`notes`。不適用的 metric 填 `null`，不要刪欄位。

正式實驗完成時，更新 `artifacts/<domain>/RESULTS_INDEX.md`；scratch 不更新 index。`paper_figures/` 只放論文或簡報定稿用的 finalized figures/tables。

## VI. Figure and Table Reading Guide

每個 artifact domain 需在 report 中說明自己的圖表：

- `rep_segmentation`：IoU/F1/count error。
- `phase_segmentation`：phase IoU、TUT MAE。
- `fatigue_rpe_vo2`：Spearman $\rho$、MAE、subject/exercise centered 指標。
- `paper_figures`：論文圖只保留最終視覺版本與來源表。

## VII. Limitations

本任務不搬移既有 artifacts。若未來要完全轉換舊 root，應另開 migration 任務，只做路徑與 index 更新。

## VIII. Reproducibility

```bash
find artifacts -maxdepth 2 -type d | sort
```

## IX. Conclusion

新 artifacts taxonomy 已定義，後續正式實驗可直接按 domain 輸出，降低搜尋與串接成本。

## References

- `docs/artifact_organization_zh.md`
