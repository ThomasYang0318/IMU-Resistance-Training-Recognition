# 015 Boundary Candidate Recall 分析

## 目的

第 015 版不是再改模型，而是回答目前 rep boundary 不準的核心問題：

```text
GT boundary 附近到底有沒有候選切點？
```

如果候選點本身不在 GT 附近，代表要換特徵；如果候選點有在附近，但最後方法沒有選到，代表瓶頸在 scoring / decoding / personalization。

## 評估設定

```text
input = datasets/workout
active blocks = 239
true internal boundaries = 2481
thresholds = +/-5, +/-10, +/-20, +/-50 samples
```

這裡只看 rep 與 rep 之間的 internal boundary；active block 的起點/終點不算，因為 014 還是 active-only 設定。

## 比較來源

候選池：

- `Raw Gyro + Energy Valleys`：gyro / energy waveform 上所有低谷，候選很密；
- `Fusion Candidate Pool`：PCA peak midpoint、多軸 midpoint、uniform priors、gyro valley、energy valley、fusion score 的聯集；
- `Multi-Axis All Midpoints`：所有軸 peak midpoint 聯集；
- `Gyro Valley at Priors`：在 autocorr/uniform prior 附近找 gyro valley；
- `Energy Valley at Priors`：在 autocorr/uniform prior 附近找 energy valley；
- `Fusion Refined Score`：014 LIFT 類的 fused score candidate；
- `Uniform Autocorr Priors`、`PCA Peak Midpoints`、`Multi-Axis Consensus`。

最終方法切割線：

- `Final 010 UGV`
- `Final 011 MFBS`
- `Final 014 STAYFIT-BA`
- `Final 014 MAXXYT-MAP`
- `Final 014 MFIT-FSTE`
- `Final 014 CARA-DTW-FS`
- `Final 014 LIFT-Fusion`

## 主要結果

| Source | Kind | Mean Candidates | Median Error | Recall +/-10 | Recall +/-20 | Recall +/-50 |
|---|---|---:|---:|---:|---:|---:|
| Raw Gyro + Energy Valleys | dense candidate pool | 306.61 | 4 | 0.9311 | 0.9964 | 1.0000 |
| Fusion Candidate Pool | dense candidate pool | 174.45 | 6 | 0.6465 | 0.8130 | 0.9669 |
| Multi-Axis All Midpoints | candidate pool | 138.70 | 9 | 0.5518 | 0.7368 | 0.9407 |
| Final 014 LIFT-Fusion | final boundary | 10.57 | 54 | 0.1435 | 0.2745 | 0.4809 |
| Final 014 MAXXYT-MAP | final boundary | 10.58 | 57 | 0.1475 | 0.2652 | 0.4575 |
| Gyro Valley at Priors | candidate pool | 10.29 | 56 | 0.1467 | 0.2608 | 0.4696 |
| Final 010 UGV | final boundary | 10.64 | 60 | 0.1399 | 0.2555 | 0.4462 |
| Final 011 MFBS | final boundary | 10.55 | 60 | 0.1181 | 0.2241 | 0.4462 |

重點解讀：

1. **訊號裡其實有候選點**
   `Raw Gyro + Energy Valleys` 在 ±20 samples 的 recall 是 `0.9964`，`Fusion Candidate Pool` 是 `0.8130`。所以不是 IMU 波形完全沒有 boundary 訊號。

2. **目前 final decoder 選不到正確候選**
   `Final 014 LIFT-Fusion` 在 ±20 samples 只有 `0.2745`，遠低於 `Fusion Candidate Pool` 的 `0.8130`。這代表目前最大的問題是 scoring / decoding，不是單純缺特徵。

3. **uniform/autocorr prior 太粗**
   `Uniform Autocorr Priors` 的 ±20 recall 只有 `0.2156`。目前方法很依賴「平均分段 + 附近找 valley」，但真實 rep duration / phase offset 有足夠變化，會把 search window 帶到錯誤位置。

4. **dense candidate pool 太密，不能直接部署**
   raw valleys 平均每個 block 有 `306.61` 個候選，雖然 recall 很高，但必須再做強 pruning / sequence decoding。

## 哪些動作最弱

以 `Final 014 LIFT-Fusion` 的 ±20 samples recall：

```text
one_arm_db_row       0.4196
db_biceps_curl       0.4116
db_squat             0.3625
db_triceps_curl      0.2500
db_rdl               0.2313
db_bench_press       0.1935
db_weighted_crunch   0.1797
db_shoulder_press    0.1444
```

`db_shoulder_press`、`db_bench_press`、`db_weighted_crunch` 還是主要弱點。

## 哪些人最弱

以 `Final 014 LIFT-Fusion` 的 ±20 samples recall：

```text
thomas0506          0.5845
kevin0509workout    0.4343
hsianshun0514workout 0.3759
yushuan0513workout  0.3101
haoyu0512workout    0.2917
yentsen0515workout  0.1704
ziho0512workout     0.1381
yoru0511workout     0.1142
yanz0510workout     0.0541
```

`yanz0510workout` 和 `yoru0511workout` 是最需要 subject-specific calibration 的人。

## 結論

目前問題不是「加更多特徵」這麼單純，而是：

```text
候選點足夠多
-> 但候選太密
-> 目前 prior / scoring 選錯
-> 導致 IoU@0.90 很低
```

下一版應該做：

1. 從 dense candidate pool 裡做 pruning，不要只在 uniform/autocorr prior 附近找一個點；
2. 用 sequence decoding / dynamic programming 在候選點之間選整組 boundary；
3. 加入 subject calibration，學個人 duration scale、boundary offset、axis weight；
4. 針對 `db_shoulder_press`、`db_bench_press` 做 exercise-specific boundary cue；
5. 評估指標保留 `within_10_samples`、`within_20_samples`、IoU@0.90 和 phase IoU。

## 輸出檔案

```text
artifacts_rep_classification/015_boundary_candidate_recall_analysis/
  015_boundary_candidate_recall_summary.csv
  015_boundary_candidate_recall_by_exercise.csv
  015_boundary_candidate_recall_by_subject.csv
  015_boundary_candidate_recall_details.csv
  015_candidate_recall_by_source.png
  015_candidate_nearest_error_by_source.png
  015_candidate_recall_exercise_within_20.png
  015_candidate_recall_subject_within_20.png
  015_key_source_error_distribution.png
```
