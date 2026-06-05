# 016 Dense Candidate Pool Dynamic Programming

> 2026-05-17 artifact cleanup note：完整 `waveform_all_sets/` 已依文件瘦身政策刪除；保留 summary、核心 CSV 與主圖。若需要全量 set waveform 圖，請用原工具重新產生。

## 目的

第 015 版證明：GT boundary 附近其實存在大量候選點，但 014 的 final decoder 選不到正確點。因此第 016 版改成：

```text
dense candidate pool
-> candidate clustering / pruning
-> dynamic programming 選整組 rep boundaries
```

這版仍是 active-only，也就是假設已知道該段是在運動中，且有 exercise hint。

## 方法

### DCP-DP

不使用少量標註，只用當下 active set 的訊號：

```text
9-axis IMU
-> PCA / multi-axis peaks
-> gyro valleys
-> energy valleys
-> autocorr period / count prior
-> cluster dense candidates
-> dynamic programming with duration constraints
```

### DCP-DP-FS

在 DCP-DP 上加入 few-shot duration calibration：

```text
每個 subject / exercise 前 3 下 labeled reps
-> median duration template
-> calibrated count estimate
-> calibrated duration constraint
-> DP 選整組 boundaries
```

## 主要結果

和既有方法比較：

| Method | Count exact | Count +/-1 | Count MAE | Rep F1@0.50 | Rep F1@0.75 | Rep F1@0.90 | Phase F1@0.50 | Phase F1@0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UGV: Universal Gyro Valley | 0.5678 | 0.8347 | 1.7034 | 0.7278 | 0.3949 | 0.1626 | 0.4552 | |
| MFBS: Multi-Feature Boundary Score | 0.4110 | 0.7797 | 1.7288 | 0.7382 | 0.4106 | 0.1621 | 0.4736 | |
| MAXXYT-MAP | 0.6822 | 0.9364 | 0.6186 | 0.7648 | 0.4380 | 0.2017 | 0.5305 | |
| LIFT-Fusion | 0.6737 | 0.9280 | 0.6314 | 0.7855 | 0.4610 | 0.1920 | 0.5319 | |
| DCP-DP | 0.6568 | 0.8856 | 1.4534 | 0.7485 | 0.4490 | 0.2010 | 0.5202 | 0.2048 |
| DCP-DP-FS | 0.6780 | 0.9280 | 0.6271 | 0.7834 | 0.4648 | 0.2051 | 0.5442 | 0.2103 |

結論：

- `DCP-DP-FS` 是目前高 IoU 與 phase split 最好的版本：
  - `Rep IoU@0.75 F1 = 0.4648`
  - `Rep IoU@0.90 F1 = 0.2051`
  - `Phase IoU@0.50 F1 = 0.5442`
- `LIFT-Fusion` 的 `IoU@0.50 F1 = 0.7855` 仍略高於 `DCP-DP-FS = 0.7834`，但差距很小。
- 沒有 few-shot calibration 的 `DCP-DP` count 較差，表示個人 duration calibration 對這批資料有實質幫助。

## 每動作 DCP-DP-FS 結果

IoU@0.75 F1：

```text
one_arm_db_row       0.6705
db_biceps_curl       0.5798
db_squat             0.5780
db_triceps_curl      0.4523
db_weighted_crunch   0.4177
db_rdl               0.3689
db_bench_press       0.3545
db_shoulder_press    0.2824
```

IoU@0.90 F1：

```text
one_arm_db_row       0.3716
db_biceps_curl       0.3349
db_squat             0.2146
db_triceps_curl      0.2085
db_rdl               0.1524
db_weighted_crunch   0.1402
db_bench_press       0.1218
db_shoulder_press    0.0920
```

`db_shoulder_press` 和 `db_bench_press` 仍是高精度 boundary 的主要弱點。

## 解讀

第 016 版證明 015 的判斷是對的：從 dense candidate pool 做 sequence decoding 可以改善高 IoU，但改善幅度仍有限。

目前卡住的地方變成：

```text
候選點有
-> DP 可以改善一點
-> 但 duration-only calibration 不足
-> 需要學 boundary offset / axis weight / exercise-specific transition cue
```

下一步可以做：

1. subject calibration 不只估 duration，也估 `boundary offset`；
2. 每個 exercise 使用不同 candidate score 權重；
3. 對 `db_shoulder_press` / `db_bench_press` 做專門錯誤圖分析；
4. 把 DP target 從 uniform duration 改成 learned phase progression；
5. 若要往模型走，應訓練 boundary scoring model，而不是直接 seq-to-seq 分類。

## 輸出檔案

```text
artifacts_rep_classification/016_dense_candidate_dp_decoder/
  016_dense_candidate_dp_comparison.csv
  016_dense_candidate_dp_comparison_with_prior.csv
  016_dense_candidate_dp_comparison_table.png
  016_dense_candidate_dp_score_bars.png
  016_dense_candidate_dp_by_exercise.csv
  016_dense_candidate_dp_by_subject.csv
  016_method_exercise_f1_iou_0.75.png
  016_method_subject_f1_iou_0.75.png
  waveform_all_sets/*.png
  methods/dcp_dp/
  methods/dcp_dp_fs/
```

`waveform_all_sets/` 共輸出 `239` 張圖。
