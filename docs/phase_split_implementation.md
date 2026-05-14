# Phase Split 實作方法與架構

## 目標

本實作的目標是在「已經切好的單一下 rep」內，精準找出
`concentric` / `eccentric` 的唯一分界點。

目前假設如下：

- 每個 `rep*.csv` 檔案已經是一個正確的 repetition；
- 每個 rep 內只會有一次 `concentric` 與 `eccentric` 的轉換；
- 任務不是重新切 rep，而是在 rep 裡面切一刀。

## 資料結構

目前流程讀取：

```text
datasets/workout/<person_session>/<exercise>/<set>/rep*.csv
```

`datasets/workout` 底下第一層資料夾會被視為 person/session ID。

例如：

```text
datasets/workout/kevin0509workout/db_bench_press/set0/rep0_173024.csv
```

會被解析成：

- person/session: `kevin0509workout`
- exercise: `db_bench_press`
- set: `set0`
- rep: `rep0_173024`

必要欄位：

- IMU: `ax`, `ay`, `az`, `gx`, `gy`, `gz`
- 真實標籤: `phase`
- 用來換算秒數誤差的時間欄位: `sensor_ts`, `host_ts`, 或 `pc_time`

## 真實切割點

真實切割點由現有 `phase` 欄位推得。

每個 rep 的處理方式：

1. 將 `phase` 正規化為小寫字串；
2. 只接受 `concentric` 與 `eccentric`；
3. 找出 `phase` 發生變化的唯一 index；
4. 若沒有變化或有多次變化，該 rep 會被略過。

概念如下：

```python
changes = np.flatnonzero(phases[1:] != phases[:-1]) + 1
true_cut = changes[0]  # only if len(changes) == 1
```

## 預測方法

工具透過 `--method` 支援多種方法。

### `midpoint`

最簡單的 baseline。

假設 phase 分界點接近 rep 中點：

```python
pred_cut = len(rep) // 2
```

如果向心與離心時間接近 50/50，這個方法會有一定穩定性，但它完全沒有學習資料差異。

### `signal`

非監督式 IMU turning-point 方法。

流程：

1. 對 IMU channel 做 robust z-score；
2. 使用 PCA 找出 dominant movement direction；
3. 對 dominant signal 做 moving average 平滑；
4. 在 rep 中央區間尋找 turning-point 候選點；
5. 用 split score 選出最佳切割點。

這個方法符合常見 wearable resistance-training signal processing 的做法，
但在目前資料上效果比監督式方法差。

### `learned-fraction`

使用訓練集統計量的 baseline。

流程：

1. 先以人為單位切 train / validation；
2. 對訓練集每個 exercise 計算：

   ```python
   true_cut / n_samples
   ```

3. 取每個 exercise 的 median cut fraction；
4. validation rep 使用該 exercise 的 fraction 進行切割。

此方法會使用訓練集標籤，但不會使用 validation 的 `phase` 標籤來預測。

### `supervised-regression`

目前主要方法。

此方法使用訓練人的資料訓練 supervised regressor，並預測未看過 validation 人的切割比例。

預測目標：

```python
target = true_cut / n_samples
```

模型：

```text
GradientBoostingRegressor
```

特徵來源：

- exercise one-hot feature；
- rep 長度與估計 sample period；
- dominant IMU waveform 統計特徵；
- waveform quantiles；
- derivative 統計特徵；
- 分段 waveform 統計；
- 非監督式 `signal` 方法預測出的 cut fraction，作為輔助特徵。

模型輸出 cut fraction 後，會轉回 sample index：

```python
pred_cut = round(pred_fraction * n_samples)
```

### Bias Correction

`supervised-regression` 可套用跨人 bias correction。

此校正只使用訓練集的人：

1. 在訓練集內做 leave-one-person-out；
2. 收集每個 exercise 的預測 residual；
3. 對每個 exercise 計算 median residual bias；
4. 對 validation 預測做小幅 shrinkage correction。

目的：改善跨人泛化，同時避免偷看 validation 標籤。

若要關閉：

```bash
--no-bias-correction
```

## 評估指標

每個 rep 會產生：

- `true_cut`
- `pred_cut`
- 絕對 sample 誤差
- 相對 rep 長度誤差
- 秒數誤差

每個人的統計包含：

- `reps`
- `mae_samples`
- `median_error_samples`
- `mae_pct_rep`
- `mae_seconds`
- `median_error_seconds`
- `acc_<=_5_samples`
- `acc_<=_10_samples`
- `acc_<=_15_samples`
- `acc_<=_5pct_rep`
- `acc_<=_10pct_rep`
- `mean_iou`
- `median_iou`
- `mean_min_iou`
- `acc_>=_75pct_iou`
- `acc_>=_90pct_iou`

目前最主要看的指標是：

```text
acc_<=_10pct_rep
```

意思是：

> 如果預測切割點與真實切割點的距離小於等於該 rep 長度的 10%，就算正確。

另外新增 phase IoU 指標，用來比較「切割後兩段區間」的重疊程度。
因為任務是在單一 rep 內切一刀，IoU 會分別計算第一段與第二段：

```text
first_iou = intersection(true first segment, predicted first segment) / union(...)
second_iou = intersection(true second segment, predicted second segment) / union(...)
mean_iou = (first_iou + second_iou) / 2
mean_min_iou = min(first_iou, second_iou) 的平均
```

`acc_>=_90pct_iou` 的意思是：

> 如果該 rep 的兩段平均 IoU 大於等於 0.90，就算 IoU 正確。

IoU 比單純 sample error 更接近 segmentation 評估，因為它直接衡量預測切割後的 phase 區間與真實 phase 區間有多少重疊。

### 每個動作的混淆矩陣

除了看切割點誤差，也會輸出每個 exercise 的 phase confusion matrix。

計算方式：

1. 使用真實 `true_cut` 將 rep 每個 sample 標成真實 phase；
2. 使用預測 `pred_cut` 將同一個 rep 每個 sample 標成預測 phase；
3. 對每個 exercise 統計 `true_phase` / `pred_phase` 的 sample 數；
4. 另外輸出該 exercise 的 sample-level phase accuracy。

輸出欄位包含：

- `exercise`
- `split`
- `true_phase`
- `pred_phase`
- `samples`
- `percent_of_true_phase`
- `exercise_sample_accuracy`

每個動作的切割正確率則在 `*_exercise_metrics.csv` 中，包含：

- `acc_<=_10pct_rep`
- `acc_>=_90pct_iou`
- `mae_seconds`
- `mean_iou`

### 使用 IoU 調整模型

`supervised-regression` 可以用訓練集內的 leave-one-person-out out-of-fold 結果，選擇 bias correction 的 shrinkage 強度。

啟用方式：

```bash
--tune-iou-bias
```

流程：

1. 只在訓練集內做 leave-one-person-out；
2. 對每一折收集 out-of-fold 預測；
3. 測試多個 bias shrink candidate：`0.0`, `0.1`, `0.25`, `0.5`, `0.75`, `1.0`；
4. 選擇訓練集 out-of-fold `mean_iou` 最高的 shrink；
5. 用選出的 shrink 修正 validation/test 預測。

這不會使用 validation/test 的真實標籤做調參，因此仍維持跨人驗證設定。

### 新使用者少量標註校正

若要回答「模型看到新人的少量標註後，能不能改善後續切割」，可以使用：

```bash
--personal-calibration-reps N
```

此模式會對每個 validation/test 人分成兩段：

1. 前 `N` 筆 rep 作為 calibration reps；
2. calibration reps 可以看真實 `phase` 切割，用來估計該使用者的 personal bias；
3. 後面的 reps 才作為真正 test reps；
4. test reps 會同時輸出未校正與個人校正後的結果，方便比較是否改善。

預設 personal bias 是全動作共用：

```bash
--personal-calibration-scope global
```

因為少量標註容易過度修正，個人 bias 會先乘上 shrink 再套用；預設是 0.25：

```bash
--personal-calibration-shrink 0.25
```

也可以改成每個 exercise 各自估 bias，沒有該 exercise calibration 樣本時退回全域 bias：

```bash
--personal-calibration-scope exercise
```

注意：這個模式會使用新人的前 `N` 筆標註，因此它衡量的是「少量個人化校正後的後續表現」，不是完全零樣本跨人泛化。

## 驗證架構

### 固定 Person Split

預設切分：

```bash
--val-ratio 0.3 --seed 42
```

目前固定切分結果：

訓練集：

- `haoyu0512workout`
- `thomas0506workout`
- `yoru0511workout`
- `yushuan0513workout`
- `ziho0512workout`

驗證集：

- `kevin0509workout`
- `yanz0510workout`

執行指令：

```bash
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --method supervised-regression \
  --person-split-output artifacts_phase_split/person_split_eval_regression_generalized \
  --val-ratio 0.3 \
  --seed 42 \
  --tune-iou-bias \
  --personal-calibration-reps 20
```

### 指定 Validation/Test 人員

可以明確指定哪些人當 validation/test，其餘所有人自動作為 training data。

例如只測 `yanz0510workout`：

```bash
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --method supervised-regression \
  --person-split-output artifacts_phase_split/test_yanz \
  --val-people yanz0510workout \
  --seed 42
```

### Leave-One-Person-Out

目前跨人驗證使用 7-fold leave-one-person-out，因為目前共有 7 個
person/session。

每一折：

- 1 個人作為 validation/test；
- 其餘 6 個人作為 training。

執行指令：

```bash
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --method supervised-regression \
  --leave-one-person-out-output artifacts_phase_split/leave_one_person_out \
  --no-bias-correction \
  --seed 42
```

這裡使用 `--no-bias-correction` 是為了讓完整輪替驗證的執行時間可控。

## 輸出檔案

固定 split 或指定 validation/test 時，輸出結構如下：

```text
artifacts_phase_split/<run_name>/
  person_split.csv
  train_person_metrics.csv
  val_person_metrics.csv
  val_calibration_person_metrics.csv
  val_test_person_metrics_uncalibrated.csv
  personal_calibration.csv
  personal_calibration_comparison.csv
  personal_calibration_iou_delta.svg
  train_exercise_metrics.csv
  val_exercise_metrics.csv
  all_exercise_metrics.csv
  train_exercise_confusion_matrix.csv
  val_exercise_confusion_matrix.csv
  all_person_metrics.csv
  val_person_accuracy_comparison.svg
  val_person_iou_comparison.svg
  all_person_accuracy_comparison.svg
  all_person_iou_comparison.svg
  regression_model_info.csv
  val_waveforms/
    plot_manifest.csv
    <person>/<exercise>/<set>/<rep>.svg
```

Leave-one-person-out 輸出：

```text
artifacts_phase_split/leave_one_person_out/
  leave_one_person_out_metrics.csv
  leave_one_person_out_exercise_metrics.csv
  leave_one_person_out_exercise_confusion_matrix.csv
  leave_one_person_out_accuracy_comparison.svg
  leave_one_person_out_iou_comparison.svg
```

## 波形切割圖

每張 waveform SVG 會顯示：

- dominant IMU waveform；
- 預測切割點：紅色垂直線；
- 真實 phase cut：黑色虛線；
- 切割後兩段 phase 區間背景；
- sample 誤差與秒數誤差。

範例輸出路徑：

```text
artifacts_phase_split/person_split_eval_regression_generalized/val_waveforms/
```

## 目前結果

### 固定 Split + Generalized Regression

Validation people：

```csv
person,acc_<=_10pct_rep,mae_seconds
kevin0509workout,91.70,0.1300
yanz0510workout,77.22,0.2414
```

### Leave-One-Person-Out

```csv
person,acc_<=_10pct_rep,mae_seconds
haoyu0512workout,75.88,0.2344
kevin0509workout,91.70,0.1357
thomas0506workout,96.73,0.0869
yanz0510workout,73.67,0.2405
yoru0511workout,71.65,0.2093
yushuan0513workout,88.52,0.1864
ziho0512workout,79.07,0.2089
```

## 已知限制

- 跨人表現差異明顯。
- `yanz0510workout`、`yoru0511workout`、`haoyu0512workout` 是目前泛化較弱的案例。
- 目前 supervised model 仍是 rep-level regression model，不是真正的 sequence segmentation model。
- 目前特徵會高度壓縮 waveform，可能遺失細部時序資訊。
- 若要更穩定，需要更多人、更多標註 rep，或更細緻的序列模型。

## 後續改善方向

可能的下一步：

- 針對每個 exercise 訓練獨立模型；
- 加入 subject-independent normalization；
- 加入 time-warped waveform features；
- 將 rep-level regression 改為逐 timestep 預測 boundary probability 的 sequence model；
- 使用 nested cross-validation 選模型，而不是固定超參數。
