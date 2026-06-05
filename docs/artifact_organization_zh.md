# 實驗產物整理規範

最後更新：2026-05-17

## 1. 目標

實驗產物要能回答三件事：

1. 這次實驗跑了什麼。
2. 輸入、程式版本與參數是什麼。
3. 結果能不能被下一次實驗引用或重跑。

每個正式實驗都要有穩定資料夾名稱、`summary.json`、可機器讀取的 metrics/tables，以及 `RESULTS_INDEX.md` 的索引紀錄。

## 2. 歷史 Root

目前保留兩個歷史 artifact root：

```text
artifacts_active_detection/
artifacts_rep_classification/
```

這兩個 root 已依論文敘事鏈瘦身，只保留核心歷史結果。後續新正式實驗不再寫入這兩個 root。

## 3. 新 Root 分類

新的正式實驗統一放在：

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

`scratch/` 只放探索與 smoke test，不進正式 index。

## 4. 正式實驗命名

```text
artifacts/<domain>/<experiment_id>_<short_slug>/
```

範例：

```text
artifacts/rep_segmentation/001_dense_candidate_dp_v1/
artifacts/fatigue_rpe_vo2/001_phase_aware_rpe_score/
```

規則：

- `experiment_id` 用三位數，同一 domain 內遞增。
- `short_slug` 使用小寫英文、數字與底線。
- 不覆蓋既有正式實驗。
- smoke test 與測參數輸出放 `artifacts/scratch/`。

## 5. 內部結構

```text
artifacts/<domain>/<experiment_id>_<short_slug>/
  summary.json
  run_config.yaml
  manifest.csv
  metrics/
  tables/
  figures/
  diagnostics/
  models/
  logs/
```

最低要求：

- `summary.json`
- 至少一個可機器讀取的結果表，例如 `metrics/overall.csv`

## 6. `summary.json` Schema

```json
{
  "schema_version": "1.0",
  "experiment_id": "001",
  "domain": "rep_segmentation",
  "name": "dense_candidate_dp_decoder_v1",
  "created_at": "2026-05-17",
  "status": "formal",
  "task": "rep segmentation",
  "question": "Can dense candidates plus DP improve rep boundary IoU?",
  "input_data": ["datasets/workout"],
  "input_artifacts": [],
  "output_dir": "artifacts/rep_segmentation/001_dense_candidate_dp_decoder_v1",
  "command": ".venv311/bin/python tools/example.py --output-dir artifacts/rep_segmentation/001_dense_candidate_dp_decoder_v1",
  "git_commit": "dirty-worktree",
  "split": "subject-wise 5-fold",
  "primary_metrics": {
    "macro_f1": null,
    "rep_iou_0_50_f1": null,
    "rep_iou_0_75_f1": null,
    "tut_mae_sec": null,
    "spearman_rho": null
  },
  "key_files": {
    "overall_metrics": "metrics/overall.csv",
    "main_figure": "figures/main_result.png"
  },
  "notes": "Short conclusion for index and downstream agents."
}
```

不適用的 metric 填 `null`。`git_commit` 若不是乾淨 commit，填 `dirty-worktree` 並在 `notes` 說明。

## 7. Index 規則

每個正式 domain 應有：

```text
artifacts/<domain>/RESULTS_INDEX.md
```

正式實驗完成或 deprecated 時必須更新 index；scratch 不更新。

每筆至少包含：

- 實驗目的。
- input data / input artifacts。
- 方法摘要。
- 執行指令。
- primary metrics。
- key files。
- 結論與限制。

## 8. Git 追蹤策略

建議追蹤：

- `summary.json`
- 小型 metrics CSV。
- 小型 table CSV。
- 代表性結果圖。
- `RESULTS_INDEX.md`

建議不追蹤：

- 大型逐 sample predictions。
- 大量 waveform examples。
- checkpoints、ONNX、raw logs。
- 可由指令重生的中間檔。

論文或簡報必要圖表才放入 `paper_figures/` 並列入 index。

## 9. Cleanup 策略

正式結果刪除是大決策，需任務報告記錄依據。2026-05-17 已依 `docs/tasks/002_artifact_cleanup_report.md` 直接刪除非保留 artifacts，並保留論文敘事鏈與最小可追溯來源資料。

後續清理原則：

- 大型可重生圖預設刪除或不追蹤。
- 保留 `summary.json`、核心 CSV、代表圖與論文圖。
- 若要遷移舊 root，另開任務，只做路徑與 index 更新，不混入演算法改動。
