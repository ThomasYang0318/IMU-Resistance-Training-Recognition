# Rep 切割文獻比較與方法架構

## 結論

目前模型還不能宣稱超越同類型高品質論文。

原因是多數文獻回報的是 repetition counting accuracy，例如一組動作的總 rep 數是否正確，或允許 +/-1 rep count error；我們目前評估的是更嚴格的 rep boundary IoU，也就是每一下 rep 的 start/end 是否切準。這兩種指標不能直接等價比較。

若只看本專案目前的 boundary-level 結果，`pca-extrema-fft` 是目前最佳方法，但 IoU@0.50 F1 仍只有 `0.2011`，代表它比 baseline 好，但還不到可以宣稱實用級精準 rep segmentation。

## 參考文獻方法

### 0. DS-MS-TCN 的結果呈現方式

Shang et al. 提出 DS-MS-TCN，使用 sequence-to-sequence temporal convolutional network 同時學 micro labels 與 macro labels。該文的 Fig. 1 / Fig. 2 以 IMU acceleration waveform 搭配陰影區塊呈現 micro label/repetition annotation；結果段落則以 sample-wise F1、segment-wise edit score、IoU F1，以及混淆矩陣比較 DS-MS-TCN、CNN、Transformer、CNN-LSTM、MS-TCN 等方法。

本專案目前尚未實作 DS-MS-TCN 深度模型，因為需要把現有 resistance-training labels 轉成 sample-wise micro/macro label sequence，再訓練 TCN。這次先參考該文「波形 + 標註陰影 + 多方法數值比較」的呈現方式，新增以下輸出：

```text
artifacts_rep_classification/waveform_method_comparison/waveform_method_comparison_<session>.png
artifacts_rep_classification/waveform_method_comparison/waveform_method_count_difference.png
artifacts_rep_classification/waveform_method_comparison/waveform_method_file_summary.csv
artifacts_rep_classification/waveform_method_comparison/sets_all/*.png
artifacts_rep_classification/waveform_method_comparison/waveform_method_all_sets_summary.csv
artifacts_rep_classification/waveform_method_comparison/set_level_results/set_level_method_average_comparison.png
artifacts_rep_classification/waveform_method_comparison/set_level_results/set_level_matched_rate_heatmap.png
artifacts_rep_classification/waveform_method_comparison/set_level_results/set_level_prediction_ratio_heatmap.png
artifacts_rep_classification/waveform_method_comparison/set_level_results/set_level_best_method_counts.png
artifacts_rep_classification/methods_comparison/rep_segmentation_methods_error_breakdown_iou_0.50.png
```

其中 waveform 圖會在同一段 IMU 波形上疊：

```text
ground truth rep intervals
dominant-axis predicted intervals
short-time-energy predicted intervals
pca-extrema predicted intervals
pca-extrema-fft predicted intervals
```

這可以直接看出各方法在同一段波形上的過切、漏切與 boundary 偏移。

另外 `sets_all/` 會為每一組 `subject / exercise / set_id` 輸出一張圖。目前共有 `210` 組 set 圖；`waveform_method_all_sets_summary.csv` 則記錄每張圖中各方法的 predicted reps、IoU >= 0.50 的 predicted reps 數量與 mean best IoU。

`set_level_results/` 則把 210 組的數值整理成結果比較圖：

- `set_level_method_average_comparison.png`：各方法平均 matched rate、mean best IoU、predicted/true reps ratio；
- `set_level_matched_rate_heatmap.png`：每組 set 的 IoU >= 0.50 matched rate；
- `set_level_prediction_ratio_heatmap.png`：每組 set 的 over-segmentation 程度；
- `set_level_best_method_counts.png`：每組最佳方法的統計。

### 1. Dominant-axis peak detection

Prabhu et al. 在 IMU rehabilitation exercise repetition counting 中，先辨識 dominant sensor axis，再用平滑後的 dominant-axis signal 做 peak detection。文中提到 signal amplitude 反映不同運動平面的動作強度，並用 mean-square 方式選 dominant axis，再以 positive/negative peaks 做 rep counting。

本專案對應實作：

```text
dominant-axis
```

差異：

- 文獻主要做 rep count；
- 本專案把 peak midpoint 轉成 rep segment boundary；
- 評估改成 IoU@0.25 / IoU@0.50 / IoU@0.75。

### 2. Short-Time Energy of acceleration magnitude

FitCoach 使用 magnitude of linear acceleration 的 short-time energy 取得細粒度 workout 資訊，並以 STE 的 local minimum 做 repetition segmentation。這類方法的直覺是：每一下動作會形成能量上升與下降，峰與峰之間的低能量點可當 boundary。

本專案對應實作：

```text
short-time-energy
```

差異：

- 文獻以 wearable mobile device 做 fitness monitoring 與 repetition analysis；
- 本專案直接用現有 IMU whole-session labels 生成 ground truth；
- 用同一份資料與同一套 IoU matching 和 subject-wise 5-fold 驗證。

### 3. PCA / dominant motion peak detection

部分 workout recognition/counting 方法會把多軸 IMU 降成一維主要運動訊號，再用 peak detection counting。PCA 的優點是不需要手動指定哪一軸是主要運動方向。

本專案對應實作：

```text
pca-extrema
```

差異：

- 不直接指定 axis；
- 用 PCA principal motion signal 找 extrema；
- 仍容易因為雜訊或 phase tremor 造成 over-segmentation。

### 4. FFT-guided PCA extrema

Zelman et al. 比較 threshold crossing、low-pass threshold、Fourier transform 等 exercise repetition counting 方法。Fourier transform 可用 dominant frequency 估計一段時間內的 rep count，但它本身無法直接給出每一下 rep 的精準 boundary。

本專案對應實作：

```text
pca-extrema-fft
```

本專案沒有單靠 FFT 切 rep，而是將 FFT 當作週期先驗：

```text
IMU 6-axis signal
-> PCA principal motion signal
-> FFT 估計 set-level dominant period
-> 用 dominant period 限制 peak distance 與候選 rep 數
-> peak / trough midpoint 形成 rep boundary
-> IoU matching 評估 boundary accuracy
```

這是目前效果最好的版本，原因是 FFT 週期約束能明顯降低 over-segmentation。

## 實驗設定

資料來源：

```text
datasets/workout
```

目前評估資料：

```text
subjects: 8
true reps: 2424
classes: 8
validation: subject-wise 5-fold
```

重點限制：

- validation subjects 不會出現在 training subjects；
- 所有方法都用同一批 ground-truth reps；
- rep segmentation 使用 greedy one-to-one IoU matching；
- 主要比較 IoU@0.50 F1，也保留 IoU@0.25 與 IoU@0.75。

## 本專案方法與 baseline 比較

| 方法 | Predicted reps | IoU@0.50 Precision | IoU@0.50 Recall | IoU@0.50 F1 | IoU@0.75 F1 |
|---|---:|---:|---:|---:|---:|
| dominant-axis | 34780 | 0.0156 | 0.2244 | 0.0292 | 0.0057 |
| short-time-energy | 21185 | 0.0362 | 0.3164 | 0.0650 | 0.0121 |
| pca-extrema | 18505 | 0.0339 | 0.2587 | 0.0599 | 0.0092 |
| pca-extrema-fft | 9618 | 0.1259 | 0.4996 | 0.2011 | 0.0593 |

比較圖：

```text
artifacts_rep_classification/methods_comparison/rep_segmentation_methods_f1.png
artifacts_rep_classification/methods_comparison/rep_segmentation_methods_iou_0.50.png
artifacts_rep_classification/methods_comparison/rep_segmentation_methods_error_breakdown_iou_0.50.png
artifacts_rep_classification/waveform_method_comparison/waveform_method_comparison_kevin0509workout_whole_session_20260509_173017.png
artifacts_rep_classification/waveform_method_comparison/waveform_method_count_difference.png
artifacts_rep_classification/waveform_method_comparison/sets_all/
artifacts_rep_classification/waveform_method_comparison/waveform_method_all_sets_summary.csv
artifacts_rep_classification/waveform_method_comparison/set_level_results/
```

## 我們方法的優點

### 1. 比一般 dominant-axis peak detection 更不依賴人工選軸

Dominant-axis 方法對手錶/IMU 的配戴方向與動作平面很敏感。`pca-extrema-fft` 先用 PCA 找主要變動方向，減少手動指定 axis 的需求。

### 2. 比 short-time-energy 更能抑制過切

STE baseline 在目前資料產生 `21185` 個 predicted reps，true reps 只有 `2424`，over-segmentation 明顯。FFT-guided 方法把 predicted reps 降到 `9618`，IoU@0.50 precision 從 `0.0362` 提升到 `0.1259`。

### 3. 比原始 PCA peak detection 更穩

原始 `pca-extrema` 的 IoU@0.50 F1 是 `0.0599`；加入 FFT dominant-period constraint 後提升到 `0.2011`。這代表週期先驗對 resistance training waveform 的切割有幫助。

### 4. 評估比單純 rep counting 更嚴格

許多文獻只看 count 是否接近，例如一組做 10 下是否算成 9、10、11 下。本專案輸出每一下 rep 的 boundary IoU，因此可以知道「有沒有切到正確位置」，而不是只知道「總數差不多」。

## 目前不足

### 1. 還沒有達到可宣稱超越文獻的程度

文獻中有不少 repetition counting 方法可達到高 counting accuracy，例如 +/-1 rep error 的 set-level accuracy 可到約 90% 或更高；但那些通常不是 boundary IoU。若要公平宣稱超越，需要同時輸出 count-level metric。

### 2. Boundary precision 仍低

即使最佳方法 `pca-extrema-fft`，predicted reps 仍是 true reps 的約 3.97 倍，false positives 很多。下一步應該加入：

```text
per-exercise period prior
subject adaptation from few labeled reps
phase-aware boundary refinement
minimum rest / reversal-point constraints
```

### 3. 沒有使用 supervised segmentation model

目前都是 classical signal processing。若要追上或超越近期模型，應該考慮用少量標註訓練 temporal boundary detector，例如 TCN / BiLSTM / 1D CNN，並用 leave-subject-out 驗證。

## 下一步建議

1. 新增 count-level metrics：
   - per-set rep count error；
   - absolute count error；
   - +/-1 rep accuracy；
   - Bland-Altman plot。

2. 新增 few-shot subject adaptation：
   - 新人前 1-2 組有標註；
   - 估計該 subject 的 period range、prominence threshold、phase duration ratio；
   - 用在後續同 subject sets。

3. 新增 phase-aware refinement：
   - 先切 rep；
   - rep 內再找 concentric/eccentric reversal point；
   - 用 boundary IoU 與 phase IoU 同時調參。

## 參考來源

- Guo et al., FitCoach / When your wearables become your fitness mate, Smart Health, 2020. https://www.sciencedirect.com/science/article/abs/pii/S2352648317300545
- Shang et al., DS-MS-TCN: Otago Exercises Recognition with a Dual-Scale Multi-Stage Temporal Convolutional Network, arXiv, 2024. https://arxiv.org/abs/2402.02910
- Prabhu et al., Recognition and Repetition Counting for Local Muscular Endurance Exercises in Exercise-Based Rehabilitation, Sensors, 2020. https://www.mdpi.com/1424-8220/20/17/4791
- Dorschky et al., LEAN: Real-Time Analysis of Resistance Training Using Wearable Computing, Sensors, 2023. https://www.mdpi.com/1424-8220/23/10/4602
- Zelman et al., Accelerometer-Based Automated Counting of Ten Exercises without Exercise-Specific Training or Tuning, Journal of Healthcare Engineering, 2020. https://doi.org/10.1155/2020/8869134
