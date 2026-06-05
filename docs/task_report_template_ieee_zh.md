# <Title>

> Task ID: `<task_id>`  
> Branch: `agent/<task_id>-<short_slug>`  
> Task Owner: `<agent_name>`  
> Last Updated: `YYYY-MM-DD`

## Abstract

用 100-200 字說明本任務解決的問題、方法、主要輸出與結論。

## Index Terms

IMU, resistance training, repetition segmentation, phase segmentation, time under tension, artifacts, task report

## I. Introduction

說明此任務在整體系統中的位置，以及為什麼需要執行。

## II. Task Definition

明確列出：

- Goal
- Inputs
- Allowed Changes
- Forbidden Changes
- Non-goals
- Outputs
- Acceptance Criteria
- Dependencies
- Handoff

## III. Input Data and Assumptions

列出使用的資料、文件、artifacts、前置任務與假設。

## IV. Method

描述方法與步驟。若有公式可用 LaTeX：

```text
$IoU = \frac{|A \cap B|}{|A \cup B|}$
```

## V. Results

列出產出的文件、artifacts、指標或清理結果。

## VI. Figure and Table Reading Guide

每張圖或表都要說明：

- 這張圖/表回答什麼問題。
- X/Y 軸、顏色、線型、threshold 的意義。
- 指標如何解讀，例如 IoU、F1、TUT MAE、Spearman $\rho$。
- 什麼結果代表成功或失敗。
- 哪些限制不應被過度解讀。

## VII. Limitations

列出本任務未處理的範圍、風險與需要主 Agent / 使用者決策的點。

## VIII. Reproducibility

列出驗收指令與重跑方式：

```bash
git status --short
```

## IX. Conclusion

用短段落說明本任務是否完成、下一步是什麼。

## References

- Project docs and artifacts index.
