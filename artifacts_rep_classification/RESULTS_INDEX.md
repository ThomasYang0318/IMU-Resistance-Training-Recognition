# Rep Segmentation / Classification Results Index

正式結果資料夾採用遞增編號：

```text
001_<experiment_name>/
002_<experiment_name>/
...
011_<experiment_name>/
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

## 005_boundary_feature_diagnostics_003_active_only

目的：

診斷第 003 版切割不準是否來自特徵選擇不足。針對每個 ground truth internal rep boundary，在附近搜尋不同特徵的 local min / max，量化候選點與真實 boundary 的 sample 誤差。

設定：

- input run：`003_active_only_pca_autocorr_refined_8class_5fold`
- search fraction：`0.35`
- min search radius：`20`
- max search radius：`160`
- features：PCA extreme、PCA velocity、acc magnitude、gyro magnitude、acc jerk、gyro jerk、motion energy、transition energy

主要觀察：

```text
overall best feature: gyro_magnitude_min
overall gyro_magnitude_min median absolute error: 36.5 samples
overall gyro_magnitude_min within 50 samples: 0.5930
db_shoulder_press best feature: transition_energy_max
db_bench_press best feature: pca_extreme_max
```

各動作建議特徵：

```text
db_bench_press: pca_extreme_max
db_biceps_curl: gyro_magnitude_min
db_rdl: pca_velocity_min
db_shoulder_press: transition_energy_max
db_squat: gyro_magnitude_min
db_triceps_curl: gyro_magnitude_min
db_weighted_crunch: gyro_magnitude_min
one_arm_db_row: gyro_magnitude_min
```

解讀：

PCA 不是所有動作的最佳 boundary 特徵。多數動作的 boundary 更接近 gyroscope magnitude minima；`db_shoulder_press` 則更接近 transition / jerk 類特徵。這表示後續應使用 exercise-aware 或 supervised boundary score，而不是所有動作共用同一個 PCA energy minima。

重點檔案：

- `boundary_feature_alignment_by_exercise.csv`
- `boundary_feature_alignment_by_subject.csv`
- `boundary_feature_alignment_overall.csv`
- `boundary_feature_recommendations_by_exercise.csv`
- `boundary_feature_median_error_by_exercise.png`
- `boundary_feature_within_50_by_exercise.png`
- `feature_waveform_examples/*.png`

## 006_active_only_pca_autocorr_feature_refined_8class_5fold

目的：

用第 005 版診斷出的 exercise-aware feature score 取代第 003 版的通用 motion-energy minima boundary refinement，驗證換特徵是否能改善 rep boundary IoU。

設定：

- segment method：`pca-autocorr-feature-refined`
- block source：`active-phase-contiguous`
- boundary refine search fraction：`0.25`
- boundary refine energy window：`51`
- classes：8
- folds：subject-wise 5-fold
- phase split：`pca-reversal`
- data：與第 003 版相同的 8 位 subject

主要數值：

```text
true reps: 2424
predicted reps: 2374
classified reps: 2338
rep IoU@0.25 F1: 0.9300
rep IoU@0.50 F1: 0.7353
rep IoU@0.75 F1: 0.3968
exercise classification accuracy: 0.8456
exercise macro F1: 0.8430
phase IoU@0.25 F1: 0.7026
phase IoU@0.50 F1: 0.4654
phase IoU@0.75 F1: 0.1957
```

與第 003 版比較：

```text
rep IoU@0.50 F1: 0.7182 -> 0.7353
rep IoU@0.75 F1: 0.3622 -> 0.3968
phase IoU@0.50 F1: 0.4383 -> 0.4654
classification accuracy: 0.8414 -> 0.8456
```

動作別重點：

```text
db_shoulder_press IoU@0.50 F1: 0.4972 -> 0.7081
db_bench_press IoU@0.50 F1: 0.5621 -> 0.5820
one_arm_db_row IoU@0.50 F1: 0.8525 -> 0.8164
db_biceps_curl IoU@0.50 F1: 0.6853 -> 0.6608
```

解讀：

換特徵方向有效，尤其大幅改善 `db_shoulder_press`；但固定 exercise feature map 也讓部分原本高分動作退步。下一步應做 train-fold 內的 per-exercise feature selection 或 supervised boundary probability，而不是手寫固定規則。

重點檔案：

- `summary.json`
- `rep_segmentation_metrics.csv`
- `rep_segmentation_metrics_by_exercise.csv`
- `rep_segmentation_metrics_by_subject.csv`
- `rep_segmentation_iou_metrics.png`
- `rep_segmentation_iou_f1_by_exercise.png`
- `rep_segmentation_iou_f1_by_subject.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `phase_split_metrics.csv`
- `phase_split_iou_metrics.png`

## 007_rep_feature_relevance_9axis_8class_5fold

目的：

分析每個 ground-truth rep 內的 9 軸 IMU 特徵，找出哪些 waveform / sensor 特徵和動作類別最穩定相關，並量化不同特徵家族是否真的有助於跨人泛化。

設定：

- input run：`003_active_only_pca_autocorr_refined_8class_5fold`
- samples：2424 ground-truth reps
- subjects：8
- exercises：8
- raw axes：`ax`, `ay`, `az`, `gx`, `gy`, `gz`, `mx`, `my`, `mz`
- extracted features：378
- validation：subject-wise 5-fold
- scoring：ANOVA F、mutual information、Random Forest importance、fold-wise top-20 stability

特徵類型：

- 單軸 time-domain：mean、std、median、min、max、range、IQR、RMS、energy、diff、slope、zero crossing、skewness、kurtosis；
- frequency：dominant frequency ratio、spectral entropy、low/mid/high band ratio；
- wavelet：Haar detail energy ratio；
- sensor norm：acc / gyro / mag / all-9 magnitude；
- PCA：acc / gyro / mag / all-9 PCA variance ratio；
- axis correlation：9 軸兩兩相關係數。

主要數值：

```text
best ablation set: acc_gyro
acc_gyro subject-wise accuracy: 0.8499 ± 0.0749
acc_only subject-wise accuracy: 0.7837 ± 0.0226
all_9_axis_features subject-wise accuracy: 0.7824 ± 0.0455
correlations_only subject-wise accuracy: 0.7078 ± 0.0976
wavelet_only subject-wise accuracy: 0.6711 ± 0.1151
mag_only subject-wise accuracy: 0.6112 ± 0.2111
gyro_only subject-wise accuracy: 0.5899 ± 0.0575
pca_only subject-wise accuracy: 0.3771 ± 0.0532
```

Top 10 overall features：

```text
axis_ax__mean
axis_ax__median
axis_ax__rms
axis_ax__energy_mean
axis_ax__abs_mean
axis_ax__max
acc_norm__abs_mean
axis_ay__median
acc_norm__mean
axis_ay__mean
```

解讀：

跨人動作辨識目前不是「九軸全部放進去」最好，而是 accelerometer + gyroscope 最穩。magnetometer 和 PCA-only 在目前資料上會拉低泛化；PCA 可以作為降噪或 visualization，但不適合單獨當主要分類特徵。動作分類最穩的特徵多為 accelerometer orientation / magnitude 類 time-domain 特徵；gyroscope 對 rep boundary refinement 仍重要，但單獨用來辨識動作不夠。

重點檔案：

- `summary.json`
- `rep_level_feature_table.csv`
- `rep_level_feature_metadata.csv`
- `rep_feature_relevance_scores.csv`
- `top_features_by_exercise.csv`
- `sensor_group_ablation_accuracy.csv`
- `sensor_group_ablation_accuracy.png`
- `top_rep_features_overall.png`
- `feature_stability_across_subjects.png`
- `feature_family_importance.png`
- `feature_importance_by_exercise.png`
- `dominant_axis_by_exercise.png`
- `exercise_feature_embedding_pca.png`
- `top20_feature_confusion_matrix.png`

## 008_feature_pair_scatter_8class

目的：

延伸第 007 版，不只看 feature ranking，而是把每個 ground-truth rep 當成一個點，使用兩個可解釋 IMU-derived features 當作 x/y 軸，直接檢查 8 個動作在二維特徵空間中的可分性。

設定：

- input run：`007_rep_feature_relevance_9axis_8class_5fold`
- samples：2424 ground-truth reps
- subjects：8
- exercises：8
- selected feature pairs：34
- scatter axes：兩個 feature 的 global z-score，只用於視覺化尺度對齊
- validation：subject-wise 5-fold
- classifier：每組 feature pair 單獨訓練 Random Forest

方法定位：

這一版是 feature separability diagnostic，不是最終模型。二維 feature scatter 的用法類似 HAR 文獻中常見的 PCA / t-SNE feature-space visualization；差別是這裡刻意使用可解釋的原始衍生特徵作為 x/y 軸。圖用來看群聚與重疊，正式判斷仍以 subject-wise accuracy、macro-F1、per-exercise F1 和 confusion matrix 為準。

主要數值：

```text
feature pairs: 34
best pair: best_acc_vs_best_spectral
feature_x: axis_ax__mean
feature_y: axis_gz__spectral_entropy
best pair accuracy: 0.7116
best pair macro-F1: 0.7122
```

Top feature pairs：

```text
best_acc_vs_best_spectral: accuracy 0.7116, macro-F1 0.7122
best_acc_vs_best_gyro: accuracy 0.6518, macro-F1 0.6517
best_acc_vs_best_wavelet: accuracy 0.6493, macro-F1 0.6451
best_acc_vs_best_corr: accuracy 0.6423, macro-F1 0.6351
acc_axis_time_top2: accuracy 0.6370, macro-F1 0.6321
```

解讀：

只用兩個 feature 已可分出部分動作，但最佳二維 pair 仍明顯低於第 007 版 `acc_gyro` 多特徵分類的 `0.8499`。這表示單一二維投影適合用來說明「哪些動作自然分開、哪些動作重疊」，但若目標是高準確率，仍需要多特徵組合與 cross-subject feature selection。

重點檔案：

- `summary.json`
- `selected_feature_pairs.csv`
- `feature_pair_metrics.csv`
- `feature_pair_per_exercise_metrics.csv`
- `feature_pair_fold_metrics.csv`
- `feature_pair_overall_scores.png`
- `feature_pair_per_exercise_f1_dotplot.png`
- `top_feature_pair_scatter_grid.png`
- `scatter_pairs/*.png`
- `confusion_matrices/*.png`

## 009_universal_rep_boundary_signal_analysis

目的：

回答「未知波形一開始還不知道動作類別時，是否已有泛化切 rep 的依據」。這版不使用 exercise label 做切割，而是用 ground-truth boundary 反推哪些 waveform 訊號適合做全動作共通的週期估計與 boundary localization。

設定：

- input run：`003_active_only_pca_autocorr_refined_8class_5fold`
- internal GT boundaries：2214
- period-estimation set rows：2898
- smooth windows：9、21、51
- energy windows：21、51、81
- boundary search radius：median rep duration × 0.35，限制在 20 到 160 samples
- boundary candidates：PCA extreme、PCA velocity、acc magnitude、gyro magnitude、acc jerk、gyro jerk、motion energy、transition energy
- period candidates：PCA motion、abs PCA motion、acc magnitude、gyro magnitude、PCA velocity、motion energy、transition energy
- period methods：autocorrelation、FFT

主要數值：

```text
best universal boundary feature: gyro_magnitude_min_s9
boundary median abs error: 36.5 samples
boundary within 50 samples: 0.5930
boundary within 100 samples: 0.8921

best period signal: pca_motion
best period method: autocorr
period median abs error: 7.0 samples
period median relative abs error: 0.0217
period within 10%: 0.8696
```

Top universal boundary features：

```text
gyro_magnitude_min_s9: median error 36.5, within50 0.5930
gyro_magnitude_min_s21: median error 39.0, within50 0.5786
gyro_magnitude_min_s51: median error 40.0, within50 0.5682
pca_velocity_min_s21_e21: median error 50.0, within50 0.5045
pca_velocity_min_s9_e21: median error 48.0, within50 0.5158
```

解讀：

目前已知道第一刀切割的合理方向：先用 `pca_motion + autocorr` 估 set 內主要週期與大致 rep count，再用 `gyro_magnitude_min_s9` 當作 universal boundary valley 做切點定位。這比直接使用第 007/008 的分類特徵更合理，因為分類特徵需要先有 rep segment。

但這還不足以達到 90% rep boundary：最佳 universal boundary feature 的 within-50-sample 比例只有 `0.5930`，弱項包含 `db_shoulder_press` within50 `0.4143`、`db_rdl` within50 `0.4888`。所以建議下一步做 `010_universal_periodic_boundary_segmenter`：用 PCA autocorr 提供週期 prior，用 gyro magnitude minima 提供 candidate boundary，再加 duration prior / dynamic programming 選出整組最合理切點。

重點檔案：

- `summary.json`
- `universal_boundary_feature_ranking.csv`
- `universal_boundary_feature_ranking.png`
- `universal_boundary_alignment_overall.csv`
- `universal_boundary_alignment_by_exercise.csv`
- `universal_boundary_within_50_by_exercise.png`
- `universal_boundary_median_error_by_exercise.png`
- `period_estimation_summary.csv`
- `period_estimation_by_exercise.csv`
- `period_estimation_error_by_signal.png`
- `period_estimation_error_by_exercise.png`
- `universal_feature_waveform_examples/*.png`

## 010_universal_periodic_gyro_valley_8class_5fold

目的：

把第 009 版找到的泛化切割方向實作成真正的 active-only rep segmenter：先用 `pca_motion + autocorrelation` 估一組裡的主要週期 / rep count，再在每個預期切點附近找 `gyro_magnitude` valley，最後用 duration prior 和 rep-count prior 選出整組切割。

設定：

- segment method：`pca-autocorr-gyro-valley`
- block source：`active-phase-contiguous`
- PCA smooth window：9
- gyro valley smooth window：9
- autocorr min period：25 samples
- autocorr max period fraction：0.8
- boundary search fraction：0.35
- rep count search radius：±2
- max reps per set：30
- classification：skipped，本版先只看 rep segmentation / phase split

主要數值：

```text
truth reps: 2720
predicted reps: 2740

rep IoU@0.25 F1: 0.9092
rep IoU@0.50 F1: 0.7278
rep IoU@0.75 F1: 0.3949
rep IoU@0.85 F1: 0.2564
rep IoU@0.90 F1: 0.1626
rep IoU@0.95 F1: 0.0670

phase IoU@0.25 F1: 0.6876
phase IoU@0.50 F1: 0.4552
phase IoU@0.75 F1: 0.1785
phase IoU@0.90 F1: 0.0432
```

每個動作 rep IoU@0.50 F1：

```text
one_arm_db_row      0.8407
db_squat            0.8413
db_weighted_crunch  0.7964
db_rdl              0.7778
db_triceps_curl     0.7720
db_shoulder_press   0.6589
db_biceps_curl      0.6133
db_bench_press      0.5499
```

解讀：

這版驗證了第 009 版的方向可以實作成可跑的切割器，整體 IoU@0.50 F1 為 `0.7278`，和第 006 版 exercise-aware feature refinement 的 `0.7353` 接近；但第 010 版第一刀不使用動作類別，因此更符合未知 waveform 的泛化流程。弱項仍是 `db_bench_press`、`db_biceps_curl` 和 `db_shoulder_press`，表示下一步需要在初切後加入分類結果，再做 second-pass exercise-aware refinement。

重點檔案：

- `summary.json`
- `rep_segmentation_metrics.csv`
- `rep_segmentation_metrics_by_exercise.csv`
- `rep_segmentation_accuracy_by_exercise_table.csv`
- `rep_segmentation_accuracy_by_exercise_table.png`
- `rep_segmentation_iou_metrics.png`
- `rep_segmentation_iou_f1_by_exercise.png`
- `rep_segmentation_iou_f1_by_subject.png`
- `phase_split_metrics.csv`
- `phase_split_iou_metrics.png`

## 010_waveform_rep_accuracy_universal_periodic_gyro_valley

目的：

針對第 010 版切割結果，輸出每一組 set 的上下排波形圖：上排顯示 ground truth rep boundary，下排顯示 predicted rep boundary，並在圖標題標示該組 IoU@0.50 precision、recall、F1 和 matched IoU。

主要數值：

```text
set count: 236
true reps: 2720
predicted reps: 2740
matched reps at IoU@0.50: 1975
set-assigned precision: 0.7208
set-assigned recall: 0.7261
set-assigned F1: 0.7234
```

重點檔案：

- `summary.json`
- `waveform_rep_accuracy_set_summary.csv`
- `waveform_rep_accuracy_by_subject.png`
- `waveform_rep_accuracy_by_exercise.png`
- `waveform_rep_accuracy_subject_exercise_heatmap.png`
- `waveform_rep_accuracy_set_f1_distribution.png`
- `sets_all/*.png`

## 011_multifeature_boundary_score_high_iou

目的：

嘗試把第 009 / 010 版的單一 `gyro_magnitude` valley 規則升級為 supervised boundary scorer。每個 active-only set 先用 `pca_motion + autocorrelation` 估主要週期與 expected rep count，再於每個預期切點附近收集多種候選點，包含 PCA extrema、acc / gyro magnitude extrema、jerk、transition energy、dominant axis extrema 等。候選點會抽局部統計特徵，使用 subject-wise GroupKFold 訓練 boundary classifier，最後以 duration prior、count prior 和 monotonic sequence constraint 選出整組 rep boundaries。

設定：

- method：`multifeature_boundary_score`
- block source：`active-phase-contiguous`
- folds：subject-wise 5-fold
- model：`logistic` (`StandardScaler + SGDClassifier(loss="log_loss")`)
- candidate top-k：1
- positive boundary radius：10 samples
- negative boundary radius：25 samples
- negative sampling ratio：6
- segmentation IoU thresholds：0.50 / 0.75 / 0.85 / 0.90 / 0.95
- phase split：`pca-reversal`

主要數值：

```text
truth reps: 2720
predicted reps: 2658

rep IoU@0.50 F1: 0.7382
rep IoU@0.75 F1: 0.4106
rep IoU@0.85 F1: 0.2510
rep IoU@0.90 F1: 0.1621
rep IoU@0.95 F1: 0.0621

median internal boundary error: 60.0 samples
median internal boundary error: 600.0 ms
within 10 samples: 0.1188
within 20 samples: 0.2246

phase IoU@0.50 F1: 0.4736
phase IoU@0.90 F1: 0.0356
```

每個動作 rep IoU@0.90 F1：

```text
db_triceps_curl     0.2600
db_biceps_curl      0.2143
db_squat            0.2079
one_arm_db_row      0.1802
db_shoulder_press   0.1700
db_rdl              0.1130
db_weighted_crunch  0.0852
db_bench_press      0.0790
```

解讀：

這版沒有達成高 IoU 目標，也沒有超過第 010 版作為下一步主方法。IoU@0.50 F1 雖然略高於第 010 版的 `0.7278`，但 IoU@0.90 F1 只有 `0.1621`，median boundary error 仍有 `60 samples / 600 ms`。這表示目前 supervised scorer 可以幫忙找「大致像 boundary」的位置，但不足以把 internal boundary 壓到 10-20 samples 內；若後續要達成 IoU@0.90 以上，需要重新處理候選點 recall、boundary label 定義、個人校正與 sequence-level alignment。

重點檔案：

- `summary.json`
- `rep_segmentation_metrics.csv`
- `rep_segmentation_metrics_by_exercise.csv`
- `rep_segmentation_metrics_by_subject.csv`
- `rep_segmentation_accuracy_by_exercise_table.csv`
- `rep_segmentation_accuracy_by_exercise_table.png`
- `rep_segmentation_iou_0.90_by_exercise_table.csv`
- `rep_segmentation_iou_0.90_by_exercise_table.png`
- `rep_segmentation_iou_metrics.png`
- `rep_segmentation_iou_f1_by_exercise.png`
- `rep_segmentation_iou_f1_by_subject.png`
- `boundary_error_overall.csv`
- `boundary_error_by_exercise.csv`
- `boundary_error_by_exercise.png`
- `phase_split_metrics.csv`
- `phase_split_iou_metrics.png`

## 011_method_comparison_high_iou

目的：

直接比較第 010 版 universal gyro-valley segmenter 和第 011 版 multi-feature boundary scorer，重點放在 IoU@0.90 高精度切割門檻。

主要數值：

```text
universal-gyro-valley IoU@0.90 F1: 0.1626
multifeature-score    IoU@0.90 F1: 0.1621

universal-gyro-valley IoU@0.50 F1: 0.7278
multifeature-score    IoU@0.50 F1: 0.7382
```

解讀：

第 011 版在 IoU@0.50 有小幅提升，但在 IoU@0.90 沒有超過第 010 版。也就是說，多特徵 scorer 對「大致分到同一個 rep」有幫助，但沒有解決高精度 boundary 對齊問題。下一步應做 candidate recall 診斷與 template / DTW alignment，而不是繼續只加局部特徵。

重點檔案：

- `rep_segmentation_methods_comparison.csv`
- `rep_segmentation_methods_comparison_by_exercise.csv`
- `rep_segmentation_methods_f1.png`
- `rep_segmentation_methods_iou_0.90.png`
- `rep_segmentation_methods_error_breakdown_iou_0.90.png`
- `rep_segmentation_exercise_delta_iou_0.90.png`

## 011_waveform_rep_accuracy_multifeature_boundary_score

目的：

針對第 011 版切割結果，輸出每一組 set 的上下排波形圖。上排是同一段 sample waveform + ground truth boundary，下排是 predicted boundary。這版使用 IoU@0.90 做 set-level 統計，專門檢查高精度 rep boundary 是否達標。

主要數值：

```text
set count: 236
true reps: 2720
predicted reps: 2658
matched reps at IoU@0.90: 436
set-assigned precision: 0.1640
set-assigned recall: 0.1603
set-assigned F1: 0.1621
```

重點檔案：

- `summary.json`
- `waveform_rep_accuracy_set_summary.csv`
- `waveform_rep_accuracy_by_subject.png`
- `waveform_rep_accuracy_by_exercise.png`
- `waveform_rep_accuracy_subject_exercise_heatmap.png`
- `waveform_rep_accuracy_set_f1_distribution.png`
- `sets_all/*.png`

## 004_waveform_rep_accuracy_003_active_only

目的：

針對第 003 版 active-only refined rep segmentation，輸出每一組 set 的波形切割圖，並在圖上直接標示該組 IoU@0.50 rep segmentation 準確率。

呈現方式：

- PCA motion waveform：深灰線；
- acceleration magnitude：淺灰線；
- 上排：同一段 sample waveform + ground truth rep boundary；
- 下排：同一段 sample waveform + predicted rep boundary；
- ground truth rep boundary：藍色；
- predicted rep boundary：紅色；
- start：實線；
- end：虛線；
- 不使用底色 shading。

主要數值：

```text
set count: 210
true reps: 2424
predicted reps: 2374
matched reps at IoU@0.50: 1711
set-assigned precision: 0.7207
set-assigned recall: 0.7059
set-assigned F1: 0.7132
```

說明：

此處數值是為了 waveform set 圖而做的 set-assigned IoU@0.50 統計；第 003 版正式 global rep IoU@0.50 F1 仍以 `003_active_only_pca_autocorr_refined_8class_5fold/summary.json` 的 `0.7182` 為準。

重點檔案：

- `summary.json`
- `waveform_rep_accuracy_set_summary.csv`
- `waveform_rep_accuracy_by_subject.png`
- `waveform_rep_accuracy_by_exercise.png`
- `waveform_rep_accuracy_subject_exercise_heatmap.png`
- `waveform_rep_accuracy_set_f1_distribution.png`
- `sets_all/*.png`

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
