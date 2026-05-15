# IMU Resistance Training Recognition

本專案目前主軸是 resistance training IMU 資料的兩段式分析：

1. 從 whole-session 波形切出每一下 repetition。
2. 將切出的 rep 以 subject-wise K-fold 驗證分類成 8 類動作，必要時可加入第 9 類 `other`。

目前分支重點不是單純訓練一個分類模型，而是比較多種 rep segmentation 方法在同一批 IMU 波形上的切割差異，並輸出 IoU、混淆矩陣、每組波形切割圖與 set-level 結果圖。

## 專案結構

```text
datasets/
  raw_data/                         早期 action classification CSV
  workout/                          whole-session workout CSV，含 subject/action/set/rep/phase 標註
models/
  inertial_student.py               XTinyHAR-style student model
preprocessing/
  window_pipeline.py                resampling、z-score、subject-wise split、sliding windows
train/
  train_student.py                  早期 action classification training entrypoint
deploy/
  export_onnx.py                    checkpoint 轉 ONNX
  luckfox_infer.py                  ONNX sliding-window inference helper
tools/
  evaluate_active_set_detection.py  active / set detection 驗證，輸出 IoU、F1、timeline 切割圖
  evaluate_active_set_window_classifier.py
                                    subject-wise window classifier active / set detection
  evaluate_rep_segmentation_classification.py
                                    rep 切割、IoU 評估、K-fold 分類、混淆矩陣輸出
  compare_rep_segmentation_iou.py   多方法 IoU 數值與圖表比較
  plot_rep_waveform_method_comparison.py
                                    每個 set 的波形切割線比較圖
  plot_set_level_method_results.py  set-level heatmap / 平均結果圖
  run_rep_project_pipeline.py       一鍵重跑目前 rep segmentation 專案流程
docs/
  rep_segmentation_classification.md
  rep_segmentation_literature_benchmark_zh.md
  project_organization_zh.md
  experiment_plan_zh.md
  change_log_zh.md
artifacts_rep_classification/
  RESULTS_INDEX.md                  rep segmentation / classification 結果版本索引
  001_active_only_labels_8class_5fold/
  002_active_only_pca_autocorr_8class_5fold/
  *_8class_5fold/                   各方法的 K-fold、IoU、混淆矩陣結果
  methods_comparison/               多方法 IoU 比較圖
  waveform_method_comparison/       波形切割圖與 set-level 結果圖
artifacts_active_detection/
  RESULTS_INDEX.md                  active / set detection 結果版本索引
  001_window_rf_action_5fold/       第 001 版 active / set detection window RF 結果
```

## 環境

建議使用專案內既有的 Python 3.11 virtualenv：

```bash
.venv311/bin/python -m pip install -r requirements.txt
```

若要重建環境：

```bash
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements.txt
```

## 一鍵重跑 Rep 專案流程

預設會重跑：

- `labels`
- `dominant-axis`
- `short-time-energy`
- `pca-extrema`
- `pca-autocorr`
- `pca-extrema-fft`

接著輸出方法比較圖、每組波形切割圖、set-level 比較圖。

```bash
.venv311/bin/python tools/run_rep_project_pipeline.py
```

先看會執行哪些指令，不真正重跑：

```bash
.venv311/bin/python tools/run_rep_project_pipeline.py --dry-run
```

如果只想重畫圖，不重跑模型與 IoU 數值：

```bash
.venv311/bin/python tools/run_rep_project_pipeline.py \
  --skip-evaluation \
  --skip-method-comparison
```

如果只想產生部分 set 的波形圖做快速檢查：

```bash
.venv311/bin/python tools/run_rep_project_pipeline.py \
  --skip-evaluation \
  --skip-method-comparison \
  --max-sets 10
```

## 驗證 Active / Set Detection

這一步只檢查「有沒有先把運動區段切出來」，不修改 rep segmentation。預設只跑可用來檢查方向的方法：

- `oracle-action`：直接用 `action_type` 非 rest，確認標註本身可形成 active set；
- `imu-hysteresis`：加速度與陀螺儀 envelope、雙閾值 hysteresis、gap merge 與最短 set 長度限制。

`imu-energy` / `imu-variance` 已驗證 segment-level 分數很低，保留為手動 baseline，不再預設執行。

```bash
.venv311/bin/python tools/evaluate_active_set_detection.py \
  --output-dir artifacts_active_detection
```

subject-wise supervised active / set detector：

```bash
.venv311/bin/python tools/evaluate_active_set_window_classifier.py \
  --output-dir artifacts_active_detection/001_window_rf_action_5fold \
  --target action
```

這個流程會用 `subject_id` 做 GroupKFold，同一個人不會同時出現在訓練與驗證。

主要輸出：

- `active_detection_metrics.csv`
- `active_detection_metrics_by_subject.csv`
- `fold_manifest.csv`
- `window_confusion_matrix.png`
- `window_rf_active_detection_f1.png`
- `timeline_examples/*.png`

目前結論是 active/rest 的 window-level 特徵可以學到，但 set-level segment IoU 仍低，表示「辨識正在動」和「切準整組 set 邊界」是兩個不同問題；後續應優先改善 set boundary post-processing 與 action label 定義。

目前正式結果版本：

- `001_window_rf_action_5fold`：subject-wise 5-fold window RF active / set detector。

## 手動執行單一方法

若要先排除休息資料，只確認「已在運動時」的 rep 切割與動作分類，使用 `--block-source active-phase-span`。這個模式直接用 `phase in {concentric,eccentric}` 的標註建立每組 set 的處理範圍，不跑 active detection。

目前正式 active-only 結果：

```bash
.venv311/bin/python tools/evaluate_rep_segmentation_classification.py \
  --data-dirs datasets/workout \
  --output-dir artifacts_rep_classification/001_active_only_labels_8class_5fold \
  --segment-method labels \
  --block-source active-phase-span \
  --folds 5 \
  --num-classes 8 \
  --evaluate-phase-split

.venv311/bin/python tools/evaluate_rep_segmentation_classification.py \
  --data-dirs datasets/workout \
  --output-dir artifacts_rep_classification/002_active_only_pca_autocorr_8class_5fold \
  --segment-method pca-autocorr \
  --block-source active-phase-span \
  --folds 5 \
  --num-classes 8 \
  --evaluate-phase-split
```

`002_active_only_pca_autocorr_8class_5fold` 目前結果：

```text
rep IoU@0.50 F1: 0.7083
exercise classification accuracy: 0.8459
phase IoU@0.50 F1: 0.4063
```

以 FFT-guided PCA extrema 為例：

```bash
.venv311/bin/python tools/evaluate_rep_segmentation_classification.py \
  --data-dirs datasets/workout \
  --output-dir artifacts_rep_classification/pca_extrema_fft_8class_5fold \
  --segment-method pca-extrema-fft \
  --folds 5 \
  --num-classes 8
```

subject-wise K-fold 使用 `subject_id` 作為 group，同一個人不會同時出現在 training 和 validation。

## 主要輸出

每個方法的輸出目錄，例如：

```text
artifacts_rep_classification/pca_extrema_fft_8class_5fold/
```

包含：

- `summary.json`
- `fold_manifest.csv`
- `classification_report.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `rep_segments_manifest.csv`
- `rep_segmentation_metrics.csv`
- `rep_segmentation_metrics_by_exercise.csv`
- `rep_segmentation_iou_metrics.png`
- `rep_segmentation_iou_f1_by_exercise.png`
- `phase_split_metrics.csv`
- `phase_split_metrics_by_phase.csv`
- `phase_split_iou_metrics.png`
- `phase_split_iou_f1_by_phase.png`

方法比較輸出：

```text
artifacts_rep_classification/methods_comparison/
```

波形切割圖輸出：

```text
artifacts_rep_classification/waveform_method_comparison/sets_all/
```

目前波形圖標示方式：

- 綠線：ground truth rep boundary
- 橘線：method predicted rep boundary
- 實線：start
- 虛線：end
- 不使用底色 shading

set-level 比較圖：

```text
artifacts_rep_classification/waveform_method_comparison/set_level_results/
```

## 目前方法

| 方法 | 說明 |
|---|---|
| `labels` | 使用資料內 `rep` + `phase` 標註，作為 oracle baseline |
| `dominant-axis` | 參考 dominant sensor axis + peak detection 類方法 |
| `short-time-energy` | 參考 acceleration magnitude short-time energy 的 rep 切割方法 |
| `pca-extrema` | 將 6-axis IMU 用 PCA 壓成主要運動訊號，再找 extrema |
| `pca-autocorr` | 用 PCA 主要運動訊號加上自相關週期估計，限制 peak distance |
| `pca-extrema-fft` | 用 FFT 估計 set-level dominant period，約束 PCA extrema 切割 |

詳細方法與文獻比較請看：

- [Rep 切割與動作分類架構](docs/rep_segmentation_classification.md)
- [Rep 切割文獻比較與方法架構](docs/rep_segmentation_literature_benchmark_zh.md)
- [專案整理說明](docs/project_organization_zh.md)
- [實驗規劃](docs/experiment_plan_zh.md)
- [變更想法紀錄](docs/change_log_zh.md)

## 早期 Student Model 流程

早期 action classification training 仍保留：

```bash
.venv311/bin/python -m train.train_student --config config.yaml
```

ONNX export：

```bash
.venv311/bin/python -m deploy.export_onnx --config config.yaml
```
