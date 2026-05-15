# 專案整理說明

## 目前專案定位

這個 repo 同時保留兩條線：

1. 早期 sliding-window action classification。
2. 目前主要的 rep segmentation + repetition-level classification。

目前實驗與交付重點是第二條線，也就是從 `datasets/workout` 的 whole-session IMU 波形切出 reps，做 subject-wise K-fold 驗證，並輸出混淆矩陣、IoU 指標與波形切割圖。

## 目錄責任

```text
datasets/
```

只放資料讀取相關程式與原始資料。`datasets/workout` 是目前 rep pipeline 的預設資料來源，但被 `.gitignore` 忽略，避免把大量原始資料誤提交。

```text
tools/
```

放實驗、評估、製圖與一鍵重跑流程。rep 專案的主要入口是：

```bash
.venv311/bin/python tools/run_rep_project_pipeline.py
```

```text
artifacts_rep_classification/
```

放目前需要展示或比較的輸出結果。雖然此目錄在 `.gitignore` 內，這個分支有刻意追蹤目前交付需要的 CSV / JSON / PNG 結果。

```text
docs/
```

放中文方法說明、文獻比較、架構與整理說明。

```text
train/ models/ preprocessing/ deploy/
```

保留早期 student model 訓練與部署流程。這些不是目前 rep segmentation 方法比較的主入口，但仍可用 `config.yaml` 跑原本的 action classification。

## 重跑順序

完整流程建議照這個順序：

1. 用 `evaluate_rep_segmentation_classification.py` 對每個 segmentation method 產生分類結果、IoU 結果與混淆矩陣。
2. 用 `compare_rep_segmentation_iou.py` 合併各方法 IoU 指標並產生比較圖。
3. 用 `plot_rep_waveform_method_comparison.py` 對每組 set 產生波形切割線圖。
4. 用 `plot_set_level_method_results.py` 將所有 set 的結果整理成 heatmap 與平均比較圖。

`run_rep_project_pipeline.py` 已經把以上四步包成同一個入口。

## 輸出規則

目前固定輸出位置：

```text
artifacts_rep_classification/<method>_8class_5fold/
artifacts_rep_classification/methods_comparison/
artifacts_rep_classification/waveform_method_comparison/
```

重要圖檔：

- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `rep_segmentation_iou_metrics.png`
- `rep_segmentation_iou_f1_by_exercise.png`
- `methods_comparison/rep_segmentation_methods_iou_0.50.png`
- `methods_comparison/rep_segmentation_methods_error_breakdown_iou_0.50.png`
- `waveform_method_comparison/sets_all/*.png`
- `waveform_method_comparison/set_level_results/*.png`

波形切割圖只畫切割線，不塗底色：

- 綠線：真實 rep 邊界；
- 橘線：預測 rep 邊界；
- 實線：start；
- 虛線：end。

## 評估原則

- 分類驗證使用 `GroupKFold`，group 是 `subject_id`。
- 同一個人不會同時出現在 training 與 validation。
- Rep segmentation 使用 interval IoU 做 one-to-one matching。
- 主要比較 `IoU >= 0.50`，並保留 `0.25` 與 `0.75`。
- 混淆矩陣與比較圖都由程式輸出，不手工繪製。

## 後續整理方向

目前沒有移動大量 artifacts，因為這些結果已經在分支中被追蹤，移動會造成很大的 diff。若之後要做更乾淨的 repo 結構，建議另開整理分支，把結果輸出拆成：

```text
artifacts/current/
artifacts/archive/
reports/
```

並決定哪些 PNG/CSV 要保留在 Git，哪些只留在本機或 release artifact。
