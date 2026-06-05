# Documentation Governance and Agent Workflow Report

> Task ID: `001`  
> Branch: `agent/001-docs-governance`  
> Task Owner: `docs-governance sub-agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務建立多 Agent 協作與文件瘦身規範，讓主 Agent 可以把工作拆給子 Agent，同時避免 README、proposal 與 todo 變成長篇流水帳。輸出包含主 Agent / 子 Agent 工作規範、文件維護政策與 IEEE 風格任務報告模板。

## Index Terms

multi-agent workflow, documentation governance, Markdown, LaTeX, IEEE report

## I. Introduction

專題已進入多任務並行階段，後續會同時整理 artifacts、規劃新實驗分類、維護論文敘事與評估 RAG/MCP。若缺少任務卡與報告格式，各 Agent 的輸出難以串接，因此需要先建立治理規範。

## II. Task Definition

- Goal：建立子 Agent 工作規範、文件瘦身政策與 IEEE 風格報告模板。
- Inputs：`README.md`、`proposal.md`、`todo.md`、`docs/README.md`、目前 artifacts 規範。
- Allowed Changes：新增 docs governance 文件與 task report。
- Forbidden Changes：不改演算法、不搬移 datasets、不重跑實驗。
- Outputs：`docs/agent_workflow_zh.md`、`docs/documentation_policy_zh.md`、`docs/task_report_template_ieee_zh.md`。
- Acceptance Criteria：主文件只需短摘要，完整任務脈絡落在 `docs/tasks/`。
- Dependencies：無。
- Handoff：後續 Agent 先讀 `docs/agent_workflow_zh.md` 與本報告，再看任務卡。

## III. Input Data and Assumptions

假設主 Agent 仍負責決策與整合，子 Agent 負責明確小任務。每個子任務使用獨立 branch，任務太大時可再拆子 Agent，但必須維持輸入、輸出、限制與驗收清楚。

## IV. Method

將工作規範拆成三層：

1. 協作層：定義主 Agent、子 Agent、分支與大決策。
2. 文件層：主文件只放短索引，細節放 task report。
3. 報告層：所有子 Agent 使用 IEEE 風格 Markdown + LaTeX 報告。
4. 資源層：只有明確任務才啟動子 Agent，並依設計/整理/盤點難度分配 reasoning 與模型成本。

## V. Results

新增文件：

- `docs/agent_workflow_zh.md`
- `docs/documentation_policy_zh.md`
- `docs/task_report_template_ieee_zh.md`

## VI. Figure and Table Reading Guide

本任務不產生實驗圖表。後續子 Agent 若產生圖表，必須在任務報告第 VI 節說明圖表問題、軸、顏色、threshold、成功/失敗判準與限制。

## VII. Limitations

本任務只建立規範，不驗證每個未來 Agent 是否完全遵守。主 Agent 合併前仍需依 checklist 審查。由於工作區已有大量既有變更，所有子 Agent 必須遵守 dirty worktree 安全規則。

## VIII. Reproducibility

```bash
sed -n '1,220p' docs/agent_workflow_zh.md
sed -n '1,180p' docs/documentation_policy_zh.md
sed -n '1,200p' docs/task_report_template_ieee_zh.md
```

## IX. Conclusion

文件治理規範已建立，後續任務可依此派工、回報與審查。

## References

- `docs/README.md`
- `docs/artifact_organization_zh.md`
