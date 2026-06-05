# IMU Resistance Training Recognition

本 repo 是「智慧穿戴阻力訓練成效評估系統」的開發工作區。系統目標是用腕戴式 IMU 完成動作辨識、repetition 切割、向心/離心 phase 切分，並計算 TUT、動作品質與疲勞相關特徵。

## 快速入口

- 專題設計：[proposal.md](proposal.md)
- 目前任務：[todo.md](todo.md)
- 文件索引：[docs/README.md](docs/README.md)
- Agent 協作規範：[docs/agent_workflow_zh.md](docs/agent_workflow_zh.md)
- 文件瘦身政策：[docs/documentation_policy_zh.md](docs/documentation_policy_zh.md)
- Artifact 規範：[docs/artifact_organization_zh.md](docs/artifact_organization_zh.md)
- IEEE 任務報告模板：[docs/task_report_template_ieee_zh.md](docs/task_report_template_ieee_zh.md)

## 開發路線

```text
Raw IMU CSV
  -> schema validation
  -> preprocessing
  -> active / set detection
  -> exercise recognition
  -> repetition segmentation
  -> concentric / eccentric phase segmentation
  -> TUT / quality / fatigue features
  -> reports
  -> App / embedded export
```

## 目前保留的結果

Artifacts 已於 2026-05-17 依「論文敘事鏈」瘦身。歷史保留集：

- `artifacts_active_detection/001_window_rf_action_5fold`
- `artifacts_rep_classification/014_literature_inspired_rep_methods`
- `artifacts_rep_classification/015_boundary_candidate_recall_analysis`
- `artifacts_rep_classification/016_dense_candidate_dp_decoder`
- `artifacts_rep_classification/017_phase_split_dcp_dp_fs`
- `artifacts_rep_classification/018_borg_gt_waveform_relation`
- `artifacts_rep_classification/019_vo2_gt_waveform_relation`
- `artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan`
- `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation`
- `artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis`
- `artifacts_rep_classification/024_imu_fatigue_component_relevance_figure`

結果索引：

- [artifacts_active_detection/RESULTS_INDEX.md](artifacts_active_detection/RESULTS_INDEX.md)
- [artifacts_rep_classification/RESULTS_INDEX.md](artifacts_rep_classification/RESULTS_INDEX.md)
- [artifacts/fatigue_rpe_vo2/RESULTS_INDEX.md](artifacts/fatigue_rpe_vo2/RESULTS_INDEX.md)
- [artifacts/paper_figures/RESULTS_INDEX.md](artifacts/paper_figures/RESULTS_INDEX.md)

後續新實驗統一輸出到：

```text
artifacts/<domain>/<experiment_id>_<short_slug>/
```

分類 root 已建立在 `artifacts/`，規則見 [docs/tasks/003_artifact_taxonomy_report.md](docs/tasks/003_artifact_taxonomy_report.md)。

## 環境

建議使用專案既有 Python 3.11 virtualenv：

```bash
.venv311/bin/python -m pip install -r requirements.txt
```

若要重建：

```bash
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements.txt
```

## 常用檢查

```bash
git status --short
find docs/tasks -maxdepth 1 -type f | sort
find artifacts_active_detection artifacts_rep_classification artifacts -maxdepth 3 -type d | sort
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```

## 工作原則

- 每次任務先定義目標、輸入、輸出、限制與驗收標準。
- 功能實作要有單元測試。
- 主文件只放短摘要與連結；完整脈絡放 `docs/tasks/`。
- 大決策需先詢問使用者，包含刪正式結果、改 artifact root、導入 RAG/MCP、改 schema、改核心指標。
