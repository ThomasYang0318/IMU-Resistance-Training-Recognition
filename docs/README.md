# 文件索引

最後更新：2026-05-17

本目錄只放專題說明、實驗紀錄、結果解讀與規範文件。根目錄的三份文件維持固定職責：

- `README.md`：給新進開發者或展示用的專案入口。
- `proposal.md`：專題目標、系統設計、模組切分與驗收原則。
- `todo.md`：目前任務拆解、優先順序與下一步。

## 文件分類

### 專案治理

- `agent_workflow_zh.md`：主 Agent / 子 Agent 分工、任務卡、分支、審查與大決策規則。
- `documentation_policy_zh.md`：README/proposal/todo/docs/tasks 的文件瘦身與維護規則。
- `task_report_template_ieee_zh.md`：每個子 Agent 任務報告的 Markdown + LaTeX IEEE 風格模板。
- `project_organization_zh.md`：repo 目錄責任、資料與程式放置原則。
- `artifact_organization_zh.md`：每次實驗產物的命名、資料夾結構與索引規範。
- `experiment_plan_zh.md`：實驗計畫與重跑順序。
- `change_log_zh.md`：重要變更紀錄。

### 核心任務文件

- `rep_segmentation_classification.md`：rep segmentation + classification 主流程。
- `rep_segmentation_literature_benchmark_zh.md`：rep segmentation 文獻與 baseline 對照。
- `ds_ms_tcn_9axis_comparison_zh.md`：9 軸 DS-MS-TCN / MS-TCN sequence model 比較。

### 編號實驗紀錄

編號實驗文件使用：

```text
<topic>_<experiment_id>_zh.md
<topic>_<experiment_id>_paper_zh.md
```

例如：

- `boundary_candidate_recall_015_zh.md`
- `dense_candidate_dp_decoder_016_zh.md`
- `phase_split_diagnostics_017_zh.md`
- `borg_waveform_relation_018_zh.md`
- `rpe_feature_correlation_020_zh.md`
- `realtime_rpe_vo2_features_022_zh.md`
- `phase_aware_fatigue_ce_rpe_023_zh.md`
- `imu_fatigue_component_relevance_024_zh.md`
- `imu_fatigue_component_relevance_024_paper_zh.md`

### 子任務報告

子 Agent 任務報告放在：

```text
docs/tasks/<task_id>_<short_slug>_report.md
```

目前初始報告：

- `tasks/001_docs_governance_report.md`
- `tasks/002_artifact_cleanup_report.md`
- `tasks/003_artifact_taxonomy_report.md`
- `tasks/004_rag_mcp_assessment_report.md`
- `tasks/005_research_framework_imu_vo2_rpe_report.md`：IMU、VO2 ground truth 與 RPE/Borg 關聯的整體研究框架設計，含文獻依據、引用對應表與 paper figure 連結。
- `tasks/006_gt_phase_imu_vo2_rpe_formal_validation_report.md`：以 LOSO nested models 正式驗證 GT phase-aware IMU、delayed VO2 與 Borg/RPE 的增益。
- `tasks/007_lowdim_set_trend_vo2_validation_report.md`：不用 CE phase-specific features，驗證低維 set-level IMU trend 與 delayed VO2 的 Borg/RPE 增益。
- `tasks/008_feature_association_evidence_table_report.md`：整理 cumulative TUT、set order、lowdim IMU trend 與 delayed VO2 對 Borg/RPE 的 feature-level 與 group-level association evidence。
- `tasks/009_controlled_one_feature_ablation_report.md`：在控制 exercise 與 within-exercise progression 後，每次只加入一個 candidate feature，檢查單一 IMU/VO2 特徵是否仍有額外增益。
- `tasks/010_module_necessity_single_feature_ranking_report.md`：開放式比較單一特徵 ranking，並用模組階梯檢查動作切分、cumulative TUT、IMU trend 與 VO2 的必要性。

## 文件撰寫規則

每份實驗文件建議固定包含：

1. 目的：這次要回答的問題。
2. 輸入：資料來源、前一版 artifact、標註版本。
3. 方法：主要演算法、模型、切分方式與參數。
4. 輸出：產物資料夾與關鍵檔案。
5. 指標：主要數值、表格、圖。
6. 結論：這次學到什麼。
7. 下一步：可執行的後續任務。

## 文件與 artifacts 對應

每個正式實驗都應同時更新：

- 對應的 `docs/*_<experiment_id>_zh.md`。
- 對應 artifact root 的 `RESULTS_INDEX.md`。
- `README.md` 中必要的入口指令或目前推薦結果。
- `todo.md` 中相關任務狀態。

如果只是探索性測參數，先放在 `artifacts/scratch/`，不寫入正式 index。

主文件只保留短摘要與連結；完整任務脈絡寫入 `docs/tasks/`，避免文件過長。
