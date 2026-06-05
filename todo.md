# Todo

最後更新：2026-05-17

## 目前原則

- 每次任務先確認目標、輸入、輸出、限制與驗收標準。
- 主文件只放短摘要與連結；完整脈絡放 `docs/tasks/`。
- 子 Agent 只有明確任務卡才啟動，並依任務難度分配 reasoning / model。
- 新正式實驗輸出到 `artifacts/<domain>/<experiment_id>_<short_slug>/`。
- subject-wise split 是模型與指標驗證的預設方式。

## 已完成

- [x] 建立專題入口與架構文件：`README.md`、`proposal.md`、`todo.md`。
- [x] 建立文件索引與 artifacts 規範：`docs/README.md`、`docs/artifact_organization_zh.md`。
- [x] 建立多 Agent 協作規範：`docs/agent_workflow_zh.md`。
- [x] 建立文件瘦身政策：`docs/documentation_policy_zh.md`。
- [x] 建立 IEEE 任務報告模板：`docs/task_report_template_ieee_zh.md`。
- [x] 建立初始任務報告：`docs/tasks/001` 到 `004`。
- [x] 依論文敘事鏈瘦身 artifacts，保留 014/015/016/017/018/019/021/022/023/024 與 active 001。
- [x] 建立後續新 artifacts 分類 root。
- [x] 建立 IMU、VO2 ground truth 與 RPE/Borg 研究框架，產出 `docs/tasks/005_research_framework_imu_vo2_rpe_report.md`。
- [x] 完成 GT phase-aware IMU + delayed VO2 對 Borg/RPE 的正式 LOSO nested validation，產出 `artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval/`。
- [x] 完成 low-dimensional non-phase set trend + delayed VO2 正式驗證，產出 `artifacts/fatigue_rpe_vo2/002_lowdim_set_trend_vo2_eval/`。
- [x] 完成 Borg/RPE feature association evidence table，產出 `artifacts/fatigue_rpe_vo2/003_feature_association_evidence_table/`。
- [x] 完成 controlled one-feature-at-a-time ablation，產出 `artifacts/fatigue_rpe_vo2/004_controlled_one_feature_ablation/`。
- [x] 完成 module necessity + open single-feature ranking，產出 `artifacts/fatigue_rpe_vo2/005_module_necessity_rpe_vo2/`。

## 下一步

1. 重新檢查 RPE 標註敘事：每個 exercise 從 1 開始時，論文主張要聚焦 within-exercise cumulative progression。
2. 設計 exercise-specific interaction / feature group ablation，避免只看單一 feature。
3. 設計 VO2 as secondary target 的章節：把 VO2 定位為可估計的延遲生理負荷，而不是目前 RPE 必要輸入。
4. 設計 real-time prefix features：每個 set 前 25/50/75/100% progress 時可用的 TUT、similarity、gyro trend。
5. 評估 predicted segmentation gap：自動 rep/set 切割後，002 的低維 features 是否仍穩定。
6. 設計 exercise-specific phase fatigue score，先用 023/024 的 heatmap 選每個動作的少量 feature。
7. 建立 `tests/` 目錄與最小 pytest 設定。
8. 實作共用 `Segment` / metrics 基礎，支援 set、rep、phase、TUT 評估。
9. 把可測試的資料 schema / loader 從 `tools/` 逐步抽出。

## 驗收指令

```bash
git status --short
find docs/tasks -maxdepth 1 -type f | sort
find artifacts_active_detection artifacts_rep_classification artifacts -maxdepth 3 -type d | sort
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```
