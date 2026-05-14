# Rep 切割與動作分類架構

## 目標

此分支的目標是建立一個完整 pipeline：

1. 從 whole-session IMU CSV 切出 repetition segments；
2. 將切出的 reps 轉成特徵；
3. 以人為單位做 K-fold 驗證，確保 validation subject 不會出現在 training；
4. 將每個 rep 分成 8 個動作類別，或額外加入第 9 類 `other`；
5. 用程式輸出 confusion matrix CSV 與 PNG 圖。

## 參考方法

目前實作優先採用 wearable IMU / resistance training 文獻中穩定常見的做法：

- 使用 accelerometer + gyroscope 的多軸統計特徵做 repetition-level classification；
- 以 subject-wise split 或 cross-validation 評估，避免同一個人的資料同時出現在 train / validation；
- 對 rep segmentation 使用平滑後 dominant motion signal 與 local extrema；
- confusion matrix 使用標準 sklearn/matplotlib pipeline 產生，而不是手工繪圖。

參考來源：

- Soro et al., *Recognition and Repetition Counting for Complex Physical Exercises with Deep Learning*, Sensors, 2019.
- LEAN, *Real-Time Analysis of Resistance Training Using Wearable Computing*, Sensors, 2023.
- ExerSense, *Real-Time Physical Exercise Segmentation, Classification, and Counting Algorithm Using an IMU Sensor*, 2020.
- *Exercise Classification in Resistance Training: A Systematic Review of Technological Approaches*, Sports Medicine, 2025.

## 資料來源

預設讀取：

```bash
datasets/workout
```

工具會尋找：

```text
*whole_session*.csv
```

必要欄位：

- IMU: `ax`, `ay`, `az`, `gx`, `gy`, `gz`
- subject: `subject_id`
- action label: `action_type`
- rep annotation: `rep`
- phase annotation: `phase`
- set: `set`

`phase` 為 `concentric` 或 `eccentric` 的區間會被視為有效 rep 內容；`none` 或 rest 區間不會被納入 rep segment。

## Rep Segmentation

目前支援兩種模式：

```bash
--segment-method labels
```

使用資料內的 `rep` + `phase` annotation 切出 reps。這是穩定 baseline，可用來評估「rep 切完後分類器能不能跨人泛化」。

```bash
--segment-method pca-extrema
```

使用 PCA dominant motion signal + smoothing + local extrema 做 experimental rep segmentation。這個方法對資料品質與 set/rest 邊界較敏感，主要用來後續和 annotation baseline 比較。

## 分類器

每個 rep 會抽取 repetition-level 特徵：

- 每個 IMU channel 的 mean/std/min/max/range/RMS/IQR；
- 一階差分的 mean absolute / std；
- accelerometer norm 與 gyroscope norm；
- dominant motion signal range；
- dominant signal turning point 數量。

分類器：

```text
StandardScaler + RandomForestClassifier
```

這是目前保守 baseline，資料量不大時比直接訓練深度模型更穩定，也方便解釋與快速驗證。

## Subject-wise K-fold

使用 `GroupKFold`，group 是 `subject_id`。

這代表：

```text
同一個 subject 永遠只會在 train 或 validation 其中一邊
```

不會發生 validation 人出現在 training set 的資料洩漏。

預設：

```bash
--folds 5
```

如果 subject 數少於指定 fold 數，程式會自動降到可行的 fold 數。

## 類別數

預設輸出 8 類：

```bash
--num-classes 8
```

如果要加入第 9 類 `other`：

```bash
--include-other
```

`other` 會吸收不在前 N 類內，或和 ground-truth rep overlap 太低的 predicted segments。

## 執行方式

8 類 baseline：

```bash
.venv311/bin/python tools/evaluate_rep_segmentation_classification.py \
  --data-dirs datasets/workout \
  --output-dir artifacts_rep_classification/labels_8class_5fold \
  --segment-method labels \
  --folds 5 \
  --num-classes 8
```

9 類 baseline：

```bash
.venv311/bin/python tools/evaluate_rep_segmentation_classification.py \
  --data-dirs datasets/workout \
  --output-dir artifacts_rep_classification/labels_9class_5fold \
  --segment-method labels \
  --folds 5 \
  --num-classes 8 \
  --include-other
```

Experimental PCA-extrema rep segmentation：

```bash
.venv311/bin/python tools/evaluate_rep_segmentation_classification.py \
  --data-dirs datasets/workout \
  --output-dir artifacts_rep_classification/pca_extrema_8class_5fold \
  --segment-method pca-extrema \
  --folds 5 \
  --num-classes 8
```

## 輸出檔案

```text
artifacts_rep_classification/<run_name>/
  summary.json
  fold_manifest.csv
  rep_segments_manifest.csv
  rep_segmentation_matches.csv
  confusion_matrix.csv
  confusion_matrix.png
  confusion_matrix_normalized.png
  classification_report.json
```

其中：

- `fold_manifest.csv`：每 fold 的 train / validation subjects；
- `confusion_matrix.csv`：矩陣原始數值；
- `confusion_matrix.png`：sklearn/matplotlib 產生的混淆矩陣圖；
- `confusion_matrix_normalized.png`：row-normalized 混淆矩陣；
- `summary.json`：整體 accuracy、macro F1、weighted F1、類別表。

## 目前 baseline 結果

已執行：

```bash
.venv311/bin/python tools/evaluate_rep_segmentation_classification.py \
  --data-dirs datasets/workout \
  --output-dir artifacts_rep_classification/labels_8class_5fold \
  --segment-method labels \
  --folds 5 \
  --num-classes 8
```

結果：

```text
truth reps: 2132
classified reps: 2132
folds: 5
accuracy: 0.8269
macro_f1: 0.8255
weighted_f1: 0.8257
```

此結果使用 annotation 切出的 reps 作為穩定 baseline，用來回答「rep 切完後，跨人動作分類能做到多少」。

9 類模式也已執行，但目前資料中沒有實際 `other` reps，因此 accuracy 與 8 類相同，macro F1 會被空的 `other` 類拉低。

Experimental `pca-extrema` 結果：

```text
truth reps: 2132
predicted reps: 15337
classified reps: 2730
accuracy: 0.7652
macro_f1: 0.7598
weighted_f1: 0.7641
```

這代表目前 PCA-extrema 切 rep 會 over-segment，後續需要針對 active set detection、peak prominence、minimum rep duration 做更細調參；暫時不建議作為主要成績。
