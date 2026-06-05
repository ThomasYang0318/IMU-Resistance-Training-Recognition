# 文件瘦身與維護政策

最後更新：2026-05-17

## 目的

文件要能支持長期開發與論文整理，但不能讓主文件變成流水帳。主文件只保留穩定入口與短摘要；完整脈絡放到 task report、experiment doc 或 change log。

## 根目錄文件分工

`README.md`：

- 專案入口。
- 目前路線。
- 重要執行指令。
- 文件索引。
- 不放完整實驗歷史。

`proposal.md`：

- 穩定系統目標。
- 資料流程。
- 模組切分。
- 評估方法。
- 不記錄每次實驗流水帳。

`todo.md`：

- 目前任務狀態。
- 下一步。
- 驗收指令。
- 不保存長期歷史。

## docs 分工

`docs/README.md` 是文件索引，只列文件用途與連結。

`docs/tasks/` 存放每個子任務的 IEEE 風格報告：

```text
docs/tasks/<task_id>_<short_slug>_report.md
```

任務完成時，只在 README/proposal/todo 或 docs index 加 1-3 行摘要與連結。

`docs/*_<experiment_id>_zh.md` 記錄正式實驗方法與結果；`docs/tasks/` 記錄 Agent 任務執行脈絡。兩者不要互相複製長段落，只用連結串接。

## 內容搬移規則

如果主文件開始出現以下情況，應搬到 task report 或 experiment doc：

- 超過 1 頁的實驗敘述。
- 大量指標表格。
- 多張圖表的解讀。
- 已完成且不再是當前任務的操作細節。
- 只對單一子任務有用的背景。

## Token 節省原則

- 優先用索引連結，不在多份文件重複同一段內容。
- `RESULTS_INDEX.md` 保存結果摘要；完整解讀放 `docs/tasks/` 或對應實驗 doc。
- 每次回顧時先讀 `docs/README.md`、`todo.md`、相關 task report，再讀大型結果 index。
- 過期內容移到 `docs/change_log_zh.md` 或 task report，不留在 README。

## RAG / MCP 觸發規則

短期不導入完整 RAG。當 `docs/tasks/` 超過約 30 份、跨文件查詢開始變慢，或論文/App 需要自動引用實驗結論時，再建立輕量本地索引：

```text
docs_index/
  manifest.json
  chunks.jsonl
  embeddings/
```

MCP 只在任務需要外部系統時啟用：

- GitHub MCP：PR、review、CI、issue、branch。
- Google Drive MCP：論文、簡報、表格進入 Drive 後的同步與整理。
- 其他 MCP：先提出用途、資料來源、權限、成本與退出方式，再詢問使用者。

## 驗收

每次文件任務完成後執行：

```bash
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```

若搜尋結果是刻意保留，必須在任務報告中說明。
