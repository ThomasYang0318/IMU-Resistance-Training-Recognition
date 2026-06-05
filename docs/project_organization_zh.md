# 專案整理說明

最後更新：2026-05-17

## 目前定位

本 repo 目前服務「智慧穿戴阻力訓練成效評估系統」：

- 從 whole-session IMU 波形切出 active set、rep 與 concentric/eccentric phase。
- 計算 TUT、品質與疲勞相關特徵。
- 以論文敘事鏈保留核心 artifacts，後續再接 App 與嵌入式部署。

## 目錄責任

```text
datasets/
```

資料讀取程式與本機資料。不要在清理 artifacts 時刪除 raw/workout datasets。

```text
tools/
```

實驗、評估、製圖與一鍵重跑流程。穩定邏輯應逐步下沉到 package module，`tools/` 保留 CLI 組合。

```text
docs/
```

專題說明、實驗紀錄、任務報告、文件治理與 artifacts 規範。索引見 `docs/README.md`。

```text
artifacts_active_detection/
artifacts_rep_classification/
```

歷史結果 root，已於 2026-05-17 依論文敘事鏈瘦身。保留清單見各自 `RESULTS_INDEX.md`。

```text
artifacts/
```

後續新正式實驗的分類 root：

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

## 文件責任

- `README.md`：短入口與重要連結。
- `proposal.md`：穩定系統設計。
- `todo.md`：目前任務與下一步。
- `docs/tasks/`：每個子 Agent 的 IEEE 風格任務報告。

主文件只放短摘要與連結，避免重複長篇實驗內容。詳細規則見 `docs/documentation_policy_zh.md`。

## Agent 工作方式

多 Agent 協作遵守 `docs/agent_workflow_zh.md`：

- 有明確任務卡才啟動子 Agent。
- 每個子任務一個 branch。
- 設計/決策任務使用較高 reasoning；單純盤點使用低成本 Agent。
- 大決策需詢問使用者。

## Artifacts 規則

後續正式實驗使用：

```text
artifacts/<domain>/<experiment_id>_<short_slug>/
```

每個正式實驗至少包含：

- `summary.json`
- 一個可機器讀取的 metrics/table
- 對應 `RESULTS_INDEX.md`
- 若由子 Agent 執行，需有 `docs/tasks/<task_id>_<slug>_report.md`

大型 waveform examples、checkpoints、raw logs 與可重生中間檔預設不追蹤。
