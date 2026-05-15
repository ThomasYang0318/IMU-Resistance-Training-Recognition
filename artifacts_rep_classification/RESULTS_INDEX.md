# Rep Segmentation / Classification Results Index

正式結果資料夾採用遞增編號：

```text
001_<experiment_name>/
002_<experiment_name>/
003_<experiment_name>/
```

## 001_active_only_labels_8class_5fold

目的：

確認只看已標註為運動中的 rep 時，動作分類模型的 oracle rep-boundary 上限，以及 phase split baseline。

設定：

- segment method：`labels`
- block source：`active-phase-span`
- classes：8
- folds：subject-wise 5-fold
- phase split：`pca-reversal`

主要數值：

```text
true reps: 2424
predicted reps: 2424
classified reps: 2424
rep IoU@0.50 F1: 1.0000
exercise classification accuracy: 0.8197
exercise macro F1: 0.8198
phase IoU@0.50 F1: 0.8333
```

重點檔案：

- `summary.json`
- `rep_segmentation_metrics.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `phase_split_metrics.csv`
- `phase_split_iou_metrics.png`

## 002_active_only_pca_autocorr_8class_5fold

目的：

拔掉 active/rest detection，直接在已標註為運動中的 set span 上評估目前 `pca-autocorr` rep segmentation、動作分類與向心/離心切割。

設定：

- segment method：`pca-autocorr`
- block source：`active-phase-span`
- classes：8
- folds：subject-wise 5-fold
- phase split：`pca-reversal`

主要數值：

```text
true reps: 2424
predicted reps: 2328
classified reps: 2290
rep IoU@0.25 F1: 0.9247
rep IoU@0.50 F1: 0.7083
rep IoU@0.75 F1: 0.3308
exercise classification accuracy: 0.8459
exercise macro F1: 0.8457
phase IoU@0.25 F1: 0.6860
phase IoU@0.50 F1: 0.4063
phase IoU@0.75 F1: 0.1671
```

解讀：

只看運動中的資料後，`pca-autocorr` rep IoU@0.50 F1 從先前 action-block 條件的約 `0.2759` 提升到 `0.7083`。這表示前段 rest / preparation contamination 是主要瓶頸之一。動作分類已達 `0.8459`，但 phase split 仍受 rep 邊界誤差影響，IoU@0.50 F1 只有 `0.4063`。

重點檔案：

- `summary.json`
- `rep_segmentation_metrics.csv`
- `rep_segmentation_metrics_by_exercise.csv`
- `rep_segmentation_metrics_by_subject.csv`
- `rep_segmentation_iou_metrics.png`
- `rep_segmentation_iou_f1_by_exercise.png`
- `rep_segmentation_iou_f1_by_subject.png`
- `classification_report.json`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `phase_split_metrics.csv`
- `phase_split_metrics_by_phase.csv`
- `phase_split_iou_metrics.png`
- `phase_split_iou_f1_by_phase.png`

## 003_active_only_pca_autocorr_refined_8class_5fold

目的：

在 active-only 條件下改善 rep boundary，避免使用 set 內部的 inactive gaps，並用局部 motion-energy minima 修正 PCA/autocorr 產生的 boundary。

設定：

- segment method：`pca-autocorr-refined`
- block source：`active-phase-contiguous`
- boundary refine search fraction：`0.25`
- boundary refine energy window：`51`
- classes：8
- folds：subject-wise 5-fold
- phase split：`pca-reversal`
- data：與 `002_active_only_pca_autocorr_8class_5fold` 相同的 8 位 subject

主要數值：

```text
true reps: 2424
predicted reps: 2374
classified reps: 2327
rep IoU@0.25 F1: 0.9287
rep IoU@0.50 F1: 0.7182
rep IoU@0.75 F1: 0.3622
exercise classification accuracy: 0.8414
exercise macro F1: 0.8382
phase IoU@0.25 F1: 0.6865
phase IoU@0.50 F1: 0.4383
phase IoU@0.75 F1: 0.1730
```

與 `002_active_only_pca_autocorr_8class_5fold` 比較：

```text
rep IoU@0.50 F1: 0.7083 -> 0.7182
rep IoU@0.75 F1: 0.3308 -> 0.3622
phase IoU@0.50 F1: 0.4063 -> 0.4383
exercise classification accuracy: 0.8459 -> 0.8414
```

解讀：

boundary refinement 有改善較嚴格的 rep boundary 指標，尤其 IoU@0.75；phase split 也小幅改善。但分類 accuracy 略降，代表 refinement 目前主要改善邊界，不一定直接改善分類特徵。

重點檔案：

- `summary.json`
- `rep_segmentation_metrics.csv`
- `rep_segmentation_metrics_by_exercise.csv`
- `rep_segmentation_iou_metrics.png`
- `rep_segmentation_iou_f1_by_exercise.png`
- `classification_report.json`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `phase_split_metrics.csv`
- `phase_split_metrics_by_phase.csv`
- `phase_split_iou_metrics.png`
- `phase_split_iou_f1_by_phase.png`
