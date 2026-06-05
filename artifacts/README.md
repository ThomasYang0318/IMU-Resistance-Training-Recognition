# Artifacts Root

最後更新：2026-05-17

後續新正式實驗輸出到此 root。歷史保留結果仍在：

- `artifacts_active_detection/`
- `artifacts_rep_classification/`

## Domains

```text
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

正式實驗使用：

```text
artifacts/<domain>/<experiment_id>_<short_slug>/
```

scratch / smoke test 放 `artifacts/scratch/`，不列入正式 index。

詳細規範見 `docs/artifact_organization_zh.md` 與 `docs/tasks/003_artifact_taxonomy_report.md`。
