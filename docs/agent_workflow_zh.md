# 主 Agent / 子 Agent 工作規範

最後更新：2026-05-17

## 目的

本文件定義「智慧穿戴阻力訓練成效評估系統」的多 Agent 協作方式。主 Agent 負責監督、拆分任務、定義串接契約與審查結果；子 Agent 負責在明確範圍內完成研究、實作、文件或 artifacts 整理。

## 角色責任

### 主 Agent

- 確認每次任務的目標、輸入、輸出、限制與驗收標準。
- 將任務拆成可平行的小任務，必要時指定依賴順序。
- 定義每個子 Agent 的任務卡與交付格式。
- 在大決策前詢問使用者。
- 審查子 Agent 交付是否符合輸入/輸出契約。
- 合併前檢查 diff、文件、artifacts、測試與任務邊界。

### 子 Agent

- 只處理任務卡指定範圍。
- 不覆蓋或回退其他 Agent / 使用者的變更。
- 若需要跨範圍修改，停止並回報主 Agent。
- 每次任務都要產出 IEEE 風格任務報告。
- 任務過大時可以再拆子 Agent，但下游子任務也必須有明確輸入、輸出、限制與驗收。

## 分支規則

每個子任務使用獨立 branch：

```text
agent/<task_id>-<short_slug>
```

範例：

```text
agent/001-docs-governance
agent/002-artifact-cleanup
agent/003-artifact-taxonomy
```

同一 branch 只處理同一任務卡。若任務範圍改變，主 Agent 重新定義任務卡。

## Dirty Worktree 安全規則

目前 repo 可能長期處於 dirty worktree。所有 Agent 必須遵守：

- 不執行 `git reset --hard`。
- 不回復非自己造成的變更。
- 不用 `git checkout -- <path>` 覆蓋檔案。
- 不刪除未列在任務卡中的資料或 artifacts。
- 若同一檔案已有他人變更，先讀懂上下文，再做最小增量修改。
- 若清理任務與既有規範衝突，任務報告必須記錄決策依據。

## 任務卡格式

每個子 Agent 開始前必須取得以下內容：

```markdown
## Task ID

## Task Owner

## Goal

## Inputs

## Allowed Changes

## Forbidden Changes

## Outputs

## Acceptance Criteria

## Dependencies

## Reporting Path
```

## 子 Agent 輸出格式

子 Agent 最終回報必須包含：

- 修改檔案。
- 產物路徑。
- schema 或 summary 變更。
- 重跑或驗收指令。
- 沒有處理的範圍。
- 風險與需要主 Agent 決策的點。
- 下一個 Agent 應讀取的 handoff 路徑。

## 串接契約

- 所有交付都要使用明確路徑，不能只用口頭描述。
- 表格輸出使用 CSV 或 JSON；圖表輸出使用 PNG。
- 每個正式 artifact 必須有 `summary.json`。
- 下游任務若依賴上游輸出，任務卡要列出精確路徑與欄位假設。
- 若輸出 schema 改變，必須更新對應 docs 與 task report。

## 平行化規則

可平行：

- 只讀研究。
- 不同文件或不同 artifact root 的整理。
- 不同模型或不同評估方法的實驗。

需排序：

- 下游需要上游輸出的任務。
- 同一份 index 或同一個 artifact root 的結構性改動。
- schema、命名規則、核心指標更動。

禁止平行：

- 多個 Agent 同時修改同一檔案且沒有主 Agent 指定合併策略。
- 多個 Agent 同時刪除或搬移同一 artifact root。
- 一個 Agent 依賴另一個尚未驗收的輸出卻先開始實作。

## Agent 智慧程度與啟動時機

子 Agent 只有在有明確任務卡時才啟動；沒有任務時不預先開 Agent，避免浪費 token。

模型 / reasoning 分配原則：

- 高智慧程度：系統設計、研究假設、方法選型、schema / interface 設計、論文敘事、RAG/MCP 架構評估。
- 中智慧程度：實作規劃、跨檔案整理、artifact 分類、結果解讀與驗收設計。
- 低成本 / 快速：單純檔案盤點、格式檢查、索引補齊、重複性清單整理。

主 Agent 派工時要在任務卡寫明：

- 任務是否需要高 reasoning。
- 是否可用低成本 Agent。
- 是否允許子 Agent 再拆子 Agent。
- 若子 Agent 無法完成，需回報阻塞點，不自行擴大任務範圍。

## 大決策

以下情況必須詢問使用者：

- 刪除正式結果。
- 修改 artifact root 或輸出 schema。
- 導入 RAG、MCP、外部服務或新資料庫。
- 改變論文敘事主線。
- 改變核心評估指標。
- 需要大量重跑實驗或長時間訓練。

## 主 Agent 合併前 Checklist

- 子 Agent 是否只改任務卡允許範圍。
- 是否有任務報告。
- 是否更新必要索引。
- artifacts 是否符合分類與 `summary.json` 規範。
- 測試或驗收指令是否執行。
- 是否留下未決標記。
- 是否影響其他子任務串接。
