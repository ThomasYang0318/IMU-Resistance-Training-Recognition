# 012 DS-MS-TCN 9 軸訓練與方法比較

> 2026-05-17 artifact cleanup note：012 sequence baseline artifacts 已依論文敘事鏈瘦身刪除；保留本文件作歷史方法脈絡參考。後續若重啟 sequence model，請輸出到 `artifacts/recognition/`。

## 目的

第 012 版把 rep segmentation 從 classical signal processing 推進到可訓練的 sequence-to-sequence baseline。方法參考 Shang et al. 的 DS-MS-TCN 呈現方式：用 sample-wise temporal segmentation、multi-stage TCN refinement、micro/macro labels、segment IoU F1 與 confusion matrix 呈現結果。

本專案和原論文的差異：

- 原論文使用 acc + gyro 6 軸；本專案使用 9 軸 `ax ay az gx gy gz mx my mz`。
- 原論文任務是 Otago exercise recognition；本專案任務是 resistance-training rep / phase segmentation 與 8 類動作辨識。
- 原論文回報 sample-wise F1、edit score、IoU F1；本專案保留 sample-wise F1 與 segment IoU F1，並額外輸出 rep boundary IoU@0.50 / 0.75 / 0.90。

參考文獻：

```text
Y. Shang, L. Beukenhorst, A. G. V. d. Heuvel, F. Dadashi, J. H. van Dieen,
and M. Pijnappels, "Otago Exercises Recognition With a Dual-Scale Multi-Stage
Temporal Convolutional Network," IEEE Journal of Biomedical and Health
Informatics, doi: 10.1109/JBHI.2024.3455426.
```

## 實作架構

新增檔案：

```text
models/ds_ms_tcn.py
tools/train_ds_ms_tcn_9axis.py
tools/compare_ds_ms_tcn_9axis.py
```

模型：

- `DS-MS-TCN`：1 個 micro SS-TCN stage + 3 個 macro SS-TCN refinement stages。
- `MS-TCN`：不使用 micro labels，只做 macro multi-stage refinement，作為 baseline。

Label 設計：

- `micro labels`：`other` + `{exercise}_{concentric/eccentric}`。
- `macro labels`：
  - exercise-only：8 類動作；
  - full-session：`other` + 8 類動作。

資料切法：

- `exercise-only`：只用 `phase in {concentric, eccentric}` 的 active samples，先確認已在運動中的 rep / phase / action 能否切準。
- `full-session`：保留 whole-session，非 active samples 標為 `other`；`other-only` windows 以固定 seed 下採樣，預設最多和 active windows 1:1。
- 驗證使用 `GroupKFold`，group 是 `subject_id`，同一個 subject 不會同時出現在 train/validation。

輸出指標：

- sample-wise macro F1；
- sample-wise micro F1；
- class-aware macro segment IoU F1@0.50 / 0.75 / 0.90；
- rep boundary IoU F1@0.50 / 0.75 / 0.90；
- phase split IoU F1@0.50 / 0.75 / 0.90；
- 每動作與每人的 rep IoU heatmap；
- macro/micro confusion matrix；
- stage timeline examples；
- 上下兩排 waveform examples：上排 ground truth、下排 prediction，只畫切割線，不塗底色。

## 執行方式

Smoke test 已跑通：

```bash
.venv311/bin/python tools/train_ds_ms_tcn_9axis.py \
  --domain exercise-only \
  --output-dir artifacts_rep_classification/012_ds_ms_tcn_9axis_smoke_exercise_only \
  --folds 2 \
  --epochs 1 \
  --batch-size 2 \
  --hidden-channels 16 \
  --num-layers 2 \
  --macro-stages 2 \
  --max-files 4 \
  --max-windows 24 \
  --plot-examples 2 \
  --example-samples 1200 \
  --model-kinds ds_ms_tcn ms_tcn
```

```bash
.venv311/bin/python tools/train_ds_ms_tcn_9axis.py \
  --domain full-session \
  --output-dir artifacts_rep_classification/012_ds_ms_tcn_9axis_smoke_full_session_other \
  --folds 2 \
  --epochs 1 \
  --batch-size 2 \
  --hidden-channels 16 \
  --num-layers 2 \
  --macro-stages 2 \
  --max-files 4 \
  --max-windows 24 \
  --plot-examples 2 \
  --example-samples 1200 \
  --model-kinds ds_ms_tcn ms_tcn
```

Smoke comparison：

```bash
.venv311/bin/python tools/compare_ds_ms_tcn_9axis.py \
  --exercise-only-dir artifacts_rep_classification/012_ds_ms_tcn_9axis_smoke_exercise_only \
  --full-session-dir artifacts_rep_classification/012_ds_ms_tcn_9axis_smoke_full_session_other \
  --output-dir artifacts_rep_classification/012_ds_ms_tcn_9axis_smoke_method_comparison
```

正式完整 5-fold：

```bash
.venv311/bin/python tools/train_ds_ms_tcn_9axis.py \
  --domain exercise-only \
  --output-dir artifacts_rep_classification/012_ds_ms_tcn_9axis_exercise_only
```

```bash
.venv311/bin/python tools/train_ds_ms_tcn_9axis.py \
  --domain full-session \
  --output-dir artifacts_rep_classification/012_ds_ms_tcn_9axis_full_session_other
```

```bash
.venv311/bin/python tools/compare_ds_ms_tcn_9axis.py
```

## 正式 5-fold 結果

正式比較已完成：

```text
exercise-only:
sessions = 9
subjects = 9
windows = 402
folds = 5
epochs = 8

full-session + other:
sessions = 14
subjects = 10
windows = 1910
folds = 5
epochs = 8
```

總比較：

| Domain | Method | Macro sample F1 | Macro segment F1@0.50 | Rep F1@0.50 | Rep F1@0.75 | Rep F1@0.90 |
|---|---|---:|---:|---:|---:|---:|
| exercise-only | DS-MS-TCN | 0.5552 | 0.3724 | 0.4765 | 0.2890 | 0.1301 |
| exercise-only | MS-TCN | 0.6349 | 0.2713 | 0.0764 | 0.0136 | 0.0023 |
| full-session + other | DS-MS-TCN | 0.6527 | 0.2814 | 0.3590 | 0.2107 | 0.0759 |
| full-session + other | MS-TCN | 0.6541 | 0.2204 | 0.0423 | 0.0119 | 0.0011 |
| classical active-only | 010 universal gyro valley |  |  | 0.7278 | 0.3949 | 0.1626 |
| classical active-only | 011 multifeature boundary score |  |  | 0.7382 | 0.4106 | 0.1621 |

結論：

- DS-MS-TCN 明顯優於 MS-TCN 的 rep boundary IoU，代表 micro label 對 rep/phase-derived boundary 有幫助。
- 但 DS-MS-TCN 仍沒有超過 010/011。最佳正式結果是 exercise-only DS-MS-TCN 的 `Rep F1@0.50 = 0.4765`、`Rep F1@0.90 = 0.1301`，低於 011 的 `0.7382 / 0.1621`。
- Full-session + other 的 macro sample F1 較高，但 rep IoU 較低，表示加入 other 後 sample-wise 分類變容易，卻沒有改善 active rep boundary。
- 目前不能宣稱 DS-MS-TCN adapted 9-axis 已超越既有 classical / boundary-scoring 方法。

正式比較圖：

```text
artifacts_rep_classification/012_ds_ms_tcn_9axis_method_comparison/012_ds_ms_tcn_vs_existing_methods.png
artifacts_rep_classification/012_ds_ms_tcn_9axis_method_comparison/012_ds_ms_tcn_method_table.png
artifacts_rep_classification/012_ds_ms_tcn_9axis_method_comparison/012_ds_ms_tcn_method_comparison.csv
```

## Smoke Test 結果

Smoke test 只用少量資料、2 folds、1 epoch、小模型，所以不能當正式準確率，只用來驗證 pipeline。

```text
exercise-only DS-MS-TCN:
macro sample F1 = 0.0411
macro segment IoU@0.50 F1 = 0.0556
rep IoU@0.50 F1 = 0.0274
rep IoU@0.90 F1 = 0.0000

exercise-only MS-TCN:
macro sample F1 = 0.1121
macro segment IoU@0.50 F1 = 0.0233
rep IoU@0.50 F1 = 0.0330
rep IoU@0.90 F1 = 0.0051

full-session DS-MS-TCN:
macro sample F1 = 0.0002
macro segment IoU@0.50 F1 = 0.0000
rep IoU@0.50 F1 = 0.0349
rep IoU@0.90 F1 = 0.0000

full-session MS-TCN:
macro sample F1 = 0.0093
macro segment IoU@0.50 F1 = 0.0000
rep IoU@0.50 F1 = 0.0000
rep IoU@0.90 F1 = 0.0000
```

這些數字很低是預期的，因為 smoke test 只訓練 1 epoch，且刻意限制資料量。正式比較應以完整 5-fold 結果為準。

## 結果檔案

每個 domain 的根目錄：

```text
run_metadata.json
summary.json
ds_ms_tcn_method_comparison.csv
ds_ms_tcn_method_comparison.png
```

每個方法子資料夾：

```text
fold_manifest.csv
fold_XX_training_history.csv
macro_confusion_matrix.png
macro_confusion_matrix_normalized.png
micro_confusion_matrix.png
micro_confusion_matrix_normalized.png
macro_segment_iou_metrics.csv
rep_segmentation_metrics.csv
rep_segmentation_metrics_by_exercise.csv
rep_segmentation_metrics_by_subject.csv
rep_segmentation_iou_metrics.png
rep_segmentation_iou_f1_by_exercise.png
rep_segmentation_iou_f1_by_subject.png
phase_split_metrics.csv
phase_split_iou_metrics.png
timeline_examples/
waveform_examples/
```

總比較資料夾：

```text
artifacts_rep_classification/012_ds_ms_tcn_9axis_method_comparison/
artifacts_rep_classification/012_ds_ms_tcn_9axis_smoke_method_comparison/
```

其中：

- `012_ds_ms_tcn_method_comparison.csv`：sequence model 與 010/011 classical rep boundary method 的數值表；
- `012_ds_ms_tcn_vs_existing_methods.png`：bar chart；
- `012_ds_ms_tcn_method_table.png`：論文式結果表。

## 解讀注意

DS-MS-TCN / MS-TCN 可以公平比較 sample-wise F1 與 macro segment IoU，因為兩者都做 sample-wise temporal segmentation。

DS-MS-TCN 和 010/011 的比較只能看 rep boundary IoU，因為 010/011 不是 sample-wise classifier。且 012 使用 9 軸，010/011 主要使用 acc+gyro 6 軸，所以文件與圖表都標註為 adapted 9-axis comparison。
