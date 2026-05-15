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
  *_8class_5fold/                   各方法的 K-fold、IoU、混淆矩陣結果
  methods_comparison/               多方法 IoU 比較圖
  waveform_method_comparison/       波形切割圖與 set-level 結果圖
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

## 手動執行單一方法

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
