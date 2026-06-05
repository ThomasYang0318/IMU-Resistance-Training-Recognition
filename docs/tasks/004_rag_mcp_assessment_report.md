# RAG and MCP Assessment Report

> Task ID: `004`  
> Branch: `agent/004-rag-mcp-assessment`  
> Task Owner: `rag-mcp-assessment sub-agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務評估目前是否需要導入 RAG 或額外 MCP。結論是短期不導入完整 RAG；目前 repo 有 docs index、RESULTS_INDEX 與 `summary.json`，搭配 `rg` 足以查找。未來當 task reports 超過約 30 份或需要跨文件問答時，再建立輕量本地 RAG。

## Index Terms

RAG, MCP, GitHub, Google Drive, documentation retrieval, experiment search

## I. Introduction

專題將累積大量任務報告與 artifacts。RAG 能加速跨文件查詢，但會增加索引維護與依賴成本。MCP 則應在需要外部系統時才開啟。

## II. Task Definition

- Goal：評估 RAG/MCP 使用時機。
- Inputs：目前 docs、artifacts index、可用插件與工作流程。
- Allowed Changes：只產生建議文件。
- Forbidden Changes：不安裝外部工具、不建立向量資料庫、不連接新服務。
- Outputs：RAG/MCP 評估報告。
- Acceptance Criteria：明確列出短期策略、觸發條件與未來結構。
- Dependencies：無。

## III. Input Data and Assumptions

目前已有：

- `docs/README.md`
- `artifacts_active_detection/RESULTS_INDEX.md`
- `artifacts_rep_classification/RESULTS_INDEX.md`
- 每個正式 artifact 的 `summary.json`

因此短期以檔案索引和 `rg` 查詢為主。

## IV. Method

評估準則：

- 查詢速度是否已成瓶頸。
- 文件量是否大到人工索引不足。
- 是否需要跨文件語意問答。
- 是否需要外部服務權限。

未來輕量 RAG 結構：

```text
docs_index/
  manifest.json
  chunks.jsonl
  embeddings/
```

## V. Results

短期策略：

- 不導入完整 RAG。
- 不新增 MCP。
- 使用 `rg`、docs index、summary.json 查詢。

觸發 RAG 條件：

- `docs/tasks/` 超過約 30 份。
- 經常需要跨實驗回答「哪個方法最好」類問題。
- 論文或 App 需要自動引用實驗結論。

MCP 使用時機：

- GitHub MCP：開 PR、查 review、CI、issue、branch。
- Google Drive MCP：論文、簡報、表格放進 Drive 並需要同步或整理。
- 其他 MCP：先提出用途、資料來源、權限與成本，再詢問使用者。

## VI. Figure and Table Reading Guide

本任務不產生圖表。若未來建立 RAG dashboard，應說明 retrieval hit rate、query latency、chunk source 與 citation 覆蓋率。

## VII. Limitations

未建立實際 RAG index，因此無法量測 retrieval latency 或 accuracy。本報告是策略判斷，不是系統實作。

## VIII. Reproducibility

```bash
rg -n "RAG|MCP|summary.json|RESULTS_INDEX" docs README.md proposal.md todo.md
```

## IX. Conclusion

目前不需要完整 RAG 或額外 MCP。先維持輕量文件索引；當文件量與跨文件問答需求上升，再建立本地 RAG。

## References

- `docs/README.md`
- `docs/artifact_organization_zh.md`
