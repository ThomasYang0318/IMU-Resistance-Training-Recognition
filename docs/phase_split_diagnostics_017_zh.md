# 017 向心 / 離心 Phase Split 診斷

> 2026-05-17 artifact cleanup note：完整 `phase_waveforms/` 已依文件瘦身政策刪除；保留 summary、phase metrics、TUT error table 與主圖。若需要全量 phase waveform 圖，請用原工具重新產生。

## 目的

第 017 版不重新切 rep，而是拿第 016 版最佳方法 `DCP-DP-FS` 的 rep boundary，測試目前向心 / 離心二分切割是否足夠支撐 TUT 分析。

這版回答的問題是：

- 在已經切好的 predicted reps 裡，把每一下再切成 concentric / eccentric，目前準度多少？
- 哪些動作和受試者的 phase split 比較弱？
- 向心 / 離心各自的 TUT 誤差大概是多少？
- 波形上看起來 GT 和 prediction 差在哪？

## 方法

輸入：

```text
artifacts_rep_classification/016_dense_candidate_dp_decoder/methods/dcp_dp_fs/rep_segments_manifest.csv
```

流程：

```text
DCP-DP-FS predicted rep segments
-> 每個 predicted rep 取 9-axis PCA principal motion signal
-> 去除首尾 trend
-> 在 rep 內 25% 到 75% 搜尋最大 reversal point
-> 依該動作 ground-truth phase order 指派 concentric / eccentric
-> 用 phase IoU 與 phase TUT error 評估
```

所有 8 個動作在目前標註中主要順序都是：

```text
concentric -> eccentric
```

## 正式結果

輸出位置：

```text
artifacts_rep_classification/017_phase_split_dcp_dp_fs/
```

整體 phase IoU：

```text
true phase segments = 5364
predicted phase segments = 5308

Phase IoU@0.50 F1 = 0.5442
Phase IoU@0.75 F1 = 0.2103
Phase IoU@0.90 F1 = 0.0469
```

分 phase：

```text
concentric IoU@0.50 F1 = 0.5683
eccentric  IoU@0.50 F1 = 0.5201
```

向心 / 離心 TUT 誤差：

```text
concentric TUT MAE = 2.3447 sec
eccentric  TUT MAE = 2.3454 sec
overall    TUT MAE = 2.3450 sec
median abs error   = 1.8062 sec
```

## 每動作 Phase IoU@0.50 F1

```text
one_arm_db_row      = 0.8271
db_biceps_curl      = 0.6808
db_squat            = 0.5803
db_triceps_curl     = 0.5731
db_weighted_crunch  = 0.5145
db_bench_press      = 0.4651
db_rdl              = 0.3868
db_shoulder_press   = 0.3027
```

`one_arm_db_row` 的 phase split 明顯最好；`db_shoulder_press` 和 `db_rdl` 最差，表示單純 PCA reversal 不足以泛化到所有動作。

## 每人 Phase IoU@0.50 F1

```text
hsianshun0514workout = 0.7342
thomas0506           = 0.6420
yushuan0513workout   = 0.6100
kevin0509workout     = 0.5805
haoyu0512workout     = 0.5421
yoru0511workout      = 0.4935
yentsen0515workout   = 0.4914
ziho0512workout      = 0.4375
yanz0510workout      = 0.3810
```

跨人差異很大，代表後續創新點應該加入 subject-specific phase offset / axis weight calibration，而不是只用全域 PCA reversal。

## 波形圖

每組 active set 都有一張上下兩排波形圖：

```text
artifacts_rep_classification/017_phase_split_dcp_dp_fs/phase_waveforms/
```

呈現方式：

- 上排：ground truth phase split；
- 下排：prediction phase split；
- 橘線：concentric；
- 綠線：eccentric；
- 不塗底色，只畫切割線與 phase tag。

## 解讀

目前 phase split 還不能說準。`Phase IoU@0.50 F1 = 54.42%` 代表大約只能抓到一半以上的向心 / 離心區間；`IoU@0.90 F1 = 4.69%` 表示高精度 phase boundary 幾乎還沒達標。

TUT 角度來看，向心 / 離心各自平均誤差約 `2.35s`。對重訓 tempo 建議而言，這個誤差偏大，尤其若單一下 rep 的 phase duration 本來只有 1 到 3 秒，就會影響節奏判斷。

## 下一步

下一版不建議只調 PCA reversal 閾值，應改成：

1. 每個動作學習 phase split feature，例如主要軸、gyro zero-crossing、velocity reversal、energy valley；
2. few-shot calibration 不只估 rep duration，也估 phase ratio 和 split offset；
3. 對 `db_shoulder_press`、`db_rdl` 做錯誤案例分析；
4. phase split 和 rep boundary 一起做 sequence decoding，避免前段 rep 邊界錯誤放大到 phase。
