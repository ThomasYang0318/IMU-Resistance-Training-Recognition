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

目前支援六種模式：

```bash
--segment-method labels
```

使用資料內的 `rep` + `phase` annotation 切出 reps。這是穩定 baseline，可用來評估「rep 切完後分類器能不能跨人泛化」。

```bash
--segment-method dominant-axis
```

用 6-axis IMU 中變異最大的單軸訊號做 peak/extrema 切割，對應常見 dominant-axis repetition counting baseline。

```bash
--segment-method short-time-energy
```

用 acceleration magnitude 的 short-time energy 找 rep 邊界，對應能量谷值切割類方法。

```bash
--segment-method pca-extrema
```

使用 PCA principal motion signal + smoothing + local extrema 做 experimental rep segmentation。這個方法比人工選軸更穩，但仍容易 over-segment。

```bash
--segment-method pca-autocorr
```

使用 PCA principal motion signal，再用 autocorrelation 估計 set 內 dominant period，並用此週期限制 peak distance。這是目前 classical signal-processing 方法中 boundary-level IoU 最好的版本。

```bash
--segment-method pca-extrema-fft
```

用 FFT 估計 set-level dominant period，再約束 PCA extrema 的 peak distance。這個版本比原始 PCA peak 穩定，但目前已被 `pca-autocorr` 超越。

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

## Rep 內九軸特徵關聯度分析

第 007 版新增的分析工具：

```bash
.venv311/bin/python tools/analyze_rep_feature_relevance.py \
  --run-dir artifacts_rep_classification/003_active_only_pca_autocorr_refined_8class_5fold \
  --output-dir artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold \
  --folds 5
```

這個工具不改 rep segmentation，而是用 ground-truth rep 片段建立特徵表，回答「每種動作在一個 rep 內到底靠哪些 IMU waveform 特徵可分」。

抽取的特徵包含：

- `ax`、`ay`、`az`、`gx`、`gy`、`gz`、`mx`、`my`、`mz` 單軸 time-domain 統計；
- FFT 類 frequency ratio / spectral entropy；
- Haar wavelet energy ratio；
- acc / gyro / mag / all-9 norm；
- acc / gyro / mag / all-9 PCA variance ratio；
- 九軸兩兩 correlation；
- rep duration。

評估方式：

- ANOVA F-score；
- mutual information；
- Random Forest feature importance；
- subject-wise GroupKFold 的 fold-wise top-20 feature stability；
- sensor / feature group ablation accuracy；
- top-feature confusion matrix。

目前第 007 版結論：

```text
acc_gyro accuracy = 0.8499
acc_only accuracy = 0.7837
all_9_axis_features accuracy = 0.7824
wavelet_only accuracy = 0.6711
mag_only accuracy = 0.6112
pca_only accuracy = 0.3771
```

因此目前動作辨識不應盲目把九軸全部塞進模型。較合理的方向是：

- 動作分類：以 accelerometer + gyroscope 的穩定 time-domain / correlation 特徵為主；
- rep boundary refinement：保留 gyroscope magnitude、transition energy、PCA extrema 等和 boundary 對齊較好的特徵；
- magnetometer：除非先做校正與 subject/device placement normalization，否則容易降低跨人泛化；
- PCA：適合做降噪、週期估計與 visualization，不適合單獨作為分類主特徵。

## Feature-pair Scatter 可分性診斷

第 008 版新增的分析工具：

```bash
.venv311/bin/python tools/analyze_feature_pair_scatter.py \
  --feature-run-dir artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold \
  --output-dir artifacts_rep_classification/008_feature_pair_scatter_8class \
  --folds 5
```

這個工具回答「如果 x 軸和 y 軸各用一個可解釋特徵，8 個動作是否能自然分開」。每個點是一個 ground-truth rep，顏色是動作類別，x/y 軸是兩個 feature 的 z-score。

為什麼可以這樣做：

- HAR / wearable sensor 文獻常用 PCA、t-SNE 或低維 feature-space scatter 來做 feature separability 的 qualitative analysis；
- 這裡不是把圖當成最終準確率，而是把它當成「特徵可分性診斷」；
- 和 PCA / t-SNE 不同，這裡刻意使用原始衍生特徵作為 x/y 軸，所以比較能解釋「到底是哪個 IMU feature 在分動作」；
- 每組 feature pair 仍會用 subject-wise GroupKFold 輸出 accuracy、macro-F1、per-exercise F1 和 confusion matrix，避免只憑視覺判斷。

目前第 008 版結論：

```text
best pair = axis_ax__mean x axis_gz__spectral_entropy
best pair accuracy = 0.7116
best pair macro-F1 = 0.7122
007 acc_gyro multi-feature accuracy = 0.8499
```

因此二維 feature-pair scatter 很適合用來寫論文中的「特徵空間可視化」與「為什麼需要多特徵組合」；但若目標是部署模型或追求 90% 以上準確率，不能只靠兩個 feature。

第 008 版的主要輸出：

- `top_feature_pair_scatter_grid.png`：最佳幾組 feature pair 的二維散點總覽；
- `scatter_pairs/*.png`：每一組 feature pair 的單張散點圖；
- `feature_pair_overall_scores.png`：每組 pair 的 accuracy / macro-F1；
- `feature_pair_per_exercise_f1_dotplot.png`：每組 pair 對每個動作的 F1；
- `confusion_matrices/*.png`：每組 pair 的混淆矩陣。

## Universal Rep Boundary 訊號分析

第 009 版新增的分析工具：

```bash
.venv311/bin/python tools/analyze_universal_rep_boundary_signals.py \
  --run-dir artifacts_rep_classification/003_active_only_pca_autocorr_refined_8class_5fold \
  --output-dir artifacts_rep_classification/009_universal_rep_boundary_signal_analysis
```

這個工具回答「未知波形還不知道是哪個動作時，到底該用什麼共通特徵先切 rep」。它分成兩個問題：

1. **週期估計**：哪個 waveform 最能估出 set 內 rep period；
2. **切點定位**：哪個 waveform 的 local min / max 最接近 ground-truth rep boundary。

目前第 009 版結論：

```text
best period signal = pca_motion
best period method = autocorr
period median relative abs error = 0.0217
period within 10% = 0.8696

best universal boundary feature = gyro_magnitude_min_s9
boundary median abs error = 36.5 samples
boundary within 50 samples = 0.5930
boundary within 100 samples = 0.8921
```

因此未知波形的第一刀建議是：

```text
active waveform
→ robust z-score acc+gyro
→ PCA principal motion signal
→ autocorrelation estimate period / expected rep count
→ search gyro magnitude valleys near period-constrained candidate boundaries
→ output preliminary rep segments
```

這裡仍不能說已經足夠準。`gyro_magnitude_min_s9` 是目前最好的 universal boundary feature，但 within-50-sample 只有 `0.5930`，代表它適合作為 candidate generator，不適合單獨當最終切割器。下一步應加上 duration prior、monotonic constraint、dynamic programming，或在初切後用分類結果做 second-pass exercise-aware refinement。

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

建議使用統一入口重跑完整流程：

```bash
.venv311/bin/python tools/run_rep_project_pipeline.py
```

先檢查會跑哪些命令：

```bash
.venv311/bin/python tools/run_rep_project_pipeline.py --dry-run
```

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

## Rep 切割正確率呈現方式

Rep segmentation 的正確率使用 IoU 呈現。每個 predicted rep segment 與 ground-truth rep segment 都是同一個 whole-session 時間軸上的 interval：

```text
IoU = overlap(predicted_rep, true_rep) / union(predicted_rep, true_rep)
```

輸出會使用 greedy one-to-one matching：先依 IoU 由高到低排序，讓每個 predicted rep 和每個 true rep 最多只被匹配一次。

目前預設輸出三個 IoU 門檻：

```text
IoU >= 0.25
IoU >= 0.50
IoU >= 0.75
```

每個門檻會輸出：

- `true_reps`
- `predicted_reps`
- `matched_reps`
- `false_positives`
- `false_negatives`
- `precision`
- `recall`
- `f1`
- `mean_matched_iou`

其中 `IoU >= 0.50` 是常見的 segment detection 合格門檻；`IoU >= 0.75` 則比較嚴格，適合看切割邊界是否接近。

另外也會輸出每個 true rep 的最佳匹配：

```text
rep_segmentation_truth_matches.csv
```

這可以用來檢查哪些 rep 沒有被切到、哪些 rep 切得偏移太多。

同時會輸出兩張程式產生的圖：

```text
rep_segmentation_iou_metrics.png
rep_segmentation_iou_f1_by_exercise.png
```

第一張呈現不同 IoU 門檻下的 precision / recall / F1；第二張呈現每個動作在不同 IoU 門檻下的 F1。

## FFT 輔助 rep 切割

部分 rep counting 文獻會用 Fourier transform 估計重複動作的 dominant frequency，再用：

```text
estimated_rep_count = dominant_frequency * set_duration
```

這類方法適合估計「一組大概做了幾下」，但單純 FFT 是全域頻率估計，無法直接給出每一下 rep 的精準 start / end。對變速、疲勞、不同人節奏不一致的 resistance training，單靠 FFT 容易失準。

因此目前實作把 FFT 當作輔助約束，而不是獨立切割器：

```text
PCA principal motion signal
-> FFT 估計 set 內 dominant period
-> 用 dominant period 約束 peak distance 與候選 rep 數
-> peak/trough midpoint 形成 rep boundary
-> 用 IoU@0.25 / IoU@0.50 / IoU@0.75 評估
```

執行 FFT-guided 版本：

```bash
python tools/evaluate_rep_segmentation_classification.py \
  --output-dir artifacts_rep_classification/pca_extrema_fft_8class_5fold \
  --segment-method pca-extrema-fft \
  --folds 5 \
  --num-classes 8
```

比較未使用 FFT 與使用 FFT：

```bash
.venv311/bin/python tools/compare_rep_segmentation_iou.py \
  --run dominant-axis=artifacts_rep_classification/dominant_axis_8class_5fold \
  --run short-time-energy=artifacts_rep_classification/short_time_energy_8class_5fold \
  --run pca-extrema=artifacts_rep_classification/pca_extrema_8class_5fold \
  --run pca-autocorr=artifacts_rep_classification/pca_autocorr_8class_5fold \
  --run pca-extrema-fft=artifacts_rep_classification/pca_extrema_fft_8class_5fold \
  --output-dir artifacts_rep_classification/methods_comparison
```

比較輸出：

```text
rep_segmentation_methods_comparison.csv
rep_segmentation_methods_comparison_by_exercise.csv
rep_segmentation_methods_f1.png
rep_segmentation_methods_iou_0.50.png
rep_segmentation_methods_error_breakdown_iou_0.50.png
```

## 輸出檔案

```text
artifacts_rep_classification/<run_name>/
  summary.json
  fold_manifest.csv
  rep_segments_manifest.csv
  rep_segmentation_matches.csv
  rep_segmentation_truth_matches.csv
  rep_segmentation_metrics.csv
  rep_segmentation_metrics_by_exercise.csv
  rep_segmentation_iou_metrics.png
  rep_segmentation_iou_f1_by_exercise.png
  confusion_matrix.csv
  confusion_matrix.png
  confusion_matrix_normalized.png
  classification_report.json
```

其中：

- `fold_manifest.csv`：每 fold 的 train / validation subjects；
- `rep_segmentation_metrics.csv`：整體 rep segmentation IoU 正確率；
- `rep_segmentation_metrics_by_exercise.csv`：每個動作的 rep segmentation IoU 正確率；
- `rep_segmentation_accuracy_by_exercise_table.csv` / `.png`：每個動作的 rep segmentation 正確率表；
- `confusion_matrix.csv`：矩陣原始數值；
- `confusion_matrix.png`：sklearn/matplotlib 產生的混淆矩陣圖；
- `confusion_matrix_normalized.png`：row-normalized 混淆矩陣；
- `summary.json`：整體 accuracy、macro F1、weighted F1、類別表。

波形圖與 set-level 圖位於：

```text
artifacts_rep_classification/waveform_method_comparison/
  waveform_method_all_sets_summary.csv
  sets_all/*.png
  set_level_results/set_level_method_average_comparison.png
  set_level_results/set_level_matched_rate_heatmap.png
  set_level_results/set_level_prediction_ratio_heatmap.png
  set_level_results/set_level_best_method_counts.png
```

波形圖只畫切割線，不塗底色：

- 綠線：ground truth rep boundary；
- 橘線：predicted rep boundary；
- 實線：start；
- 虛線：end。

## 目前方法比較結果

目前 `methods_comparison` 的 boundary-level IoU 結果如下：

| 方法 | Predicted reps | IoU@0.50 Precision | IoU@0.50 Recall | IoU@0.50 F1 | IoU@0.75 F1 |
|---|---:|---:|---:|---:|---:|
| dominant-axis | 22103 | 0.0362 | 0.3300 | 0.0652 | 0.0101 |
| short-time-energy | 15391 | 0.0670 | 0.4253 | 0.1157 | 0.0231 |
| pca-extrema | 11480 | 0.0841 | 0.3981 | 0.1388 | 0.0237 |
| pca-autocorr | 4846 | 0.2070 | 0.4138 | 0.2759 | 0.0930 |
| pca-extrema-fft | 5942 | 0.1439 | 0.3527 | 0.2044 | 0.0304 |

`pca-autocorr` 的 precision、F1 與 over-segmentation 控制目前最好。它把 predicted reps 從 `pca-extrema-fft` 的 `5942` 降到 `4846`，IoU@0.50 F1 從 `0.2044` 提升到 `0.2759`。但 boundary-level F1 仍不高，不能宣稱已超越高品質文獻。

## Oracle Label Baseline

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
rep segmentation IoU@0.50 F1: 1.0000
```

此結果使用 annotation 切出的 reps 作為穩定 baseline，用來回答「rep 切完後，跨人動作分類能做到多少」。

9 類模式也已執行，但目前資料中沒有實際 `other` reps，因此 accuracy 與 8 類相同，macro F1 會被空的 `other` 類拉低。

## 第 010 版：Universal Periodic Gyro-Valley Segmenter

第 010 版把第 009 版的分析結果實作成可評估的切割器。流程是：

```text
active-only set span
-> PCA motion signal
-> autocorrelation 估主要 rep period / expected rep count
-> 在 expected boundary 附近搜尋 gyro magnitude valley
-> 用 duration prior + rep-count prior 選整組切點
-> 用 IoU@0.25 / 0.50 / 0.75 評估 rep segmentation
-> 額外輸出 phase split IoU 和每組波形切割圖
```

執行指令：

```bash
.venv311/bin/python tools/evaluate_rep_segmentation_classification.py \
  --data-dirs datasets/workout \
  --output-dir artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold \
  --segment-method pca-autocorr-gyro-valley \
  --block-source active-phase-contiguous \
  --num-classes 8 \
  --folds 5 \
  --min-segment-samples 20 \
  --smooth-window 9 \
  --gyro-valley-smooth-window 9 \
  --autocorr-min-period-samples 25 \
  --autocorr-max-period-fraction 0.8 \
  --boundary-refine-search-fraction 0.35 \
  --periodic-count-search-radius 2 \
  --periodic-max-reps 30 \
  --evaluate-phase-split \
  --phase-split-method pca-reversal \
  --skip-classification
```

波形圖：

```bash
.venv311/bin/python tools/plot_waveform_rep_accuracy.py \
  --run-dir artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold \
  --output-dir artifacts_rep_classification/010_waveform_rep_accuracy_universal_periodic_gyro_valley \
  --iou-threshold 0.5 \
  --min-set-reps 1
```

主要結果：

```text
rep IoU@0.25 F1 = 0.9092
rep IoU@0.50 F1 = 0.7278
rep IoU@0.75 F1 = 0.3949

phase IoU@0.50 F1 = 0.4552
waveform set-level IoU@0.50 F1 = 0.7234
```

每個動作的正確率表在：

```text
artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold/rep_segmentation_accuracy_by_exercise_table.png
```
