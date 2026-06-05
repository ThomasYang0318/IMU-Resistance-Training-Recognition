# 變更想法紀錄

此文件用來記錄每次實作前的想法、假設、決策與結果。目的不是取代 git commit，而是留下「為什麼這樣改」。

## 使用規則

每次實作前先新增一筆「預計變更」。使用者審核後再動手。實作完成後補上「實際結果」與「下一步」。

每筆變更至少包含：

```text
日期：
狀態：
目的：
背景問題：
假設：
預計改動：
不做的事：
預期改善：
評估指標：
風險：
審核結果：
實際結果：
下一步：
```

狀態可用：

- `proposal`：提出計畫，等待審核；
- `approved`：已核准，準備實作；
- `implemented`：已實作；
- `rejected`：不做；
- `superseded`：被後續方案取代。

## 2026-05-17：024 圖論文撰寫版

日期：2026-05-17

狀態：implemented

目的：

將第 024 版 IMU fatigue component relevance 圖整理成可直接放入論文的章節，詳細說明為什麼要跑這張圖、方法如何設計、結果如何解讀，以及引用哪些文獻支撐。

實際改動：

- 新增 `docs/imu_fatigue_component_relevance_024_paper_zh.md`；
- 在第 024 版簡短文件與結果索引中加入論文撰寫版連結；
- 補上 Introduction、Methods、Results、Discussion、Limitations、figure caption 與 IEEE 格式引用。

結論：

第 024 圖應定位成「建模前的特徵合理性驗證」，用來支撐 IMU 可量化 fatigue-related movement changes，而不是宣稱 IMU 可直接量測肌肉疲勞。後續模型應採用 exercise-aware、phase-aware 與 subject calibration 設計。

## 2026-05-16：024 IMU 疲勞相關成分結果圖

日期：2026-05-16

狀態：implemented

目的：

產生一張可放入論文或簡報的結果圖，用數據說明哪些 IMU / VO2 成分和 Borg/RPE 有關。這張圖用來支撐「IMU 不能直接量測肌肉疲勞，但可以量化疲勞相關動作學變化」這個研究定位。

實際改動：

- 新增 `tools/plot_imu_fatigue_component_relevance.py`；
- 新增 `docs/imu_fatigue_component_relevance_024_zh.md`；
- 合併第 023 版 CE phase-aware correlation 與第 022 版 VO2 correlation；
- 輸出 correlation bar chart、raw vs within-subject/exercise comparison、exercise-feature heatmap。

正式結果：

```text
output = artifacts_rep_classification/024_imu_fatigue_component_relevance_figure/

Accumulated TUT        rho =  0.4594
Delayed VO2 slope     rho =  0.3639
VO2 baseline delta    rho = -0.3500
CE phase range        rho =  0.3377
CE phase similarity   rho = -0.3272
Concentric gyro       rho =  0.2830
Phase movement rate   rho =  0.2801
Phase timing drift    rho =  0.1740
CE ratio drift        rho =  0.1011
```

結論：

結果支持 IMU 可量化疲勞相關動作學表徵，其中最明顯的是累積 TUT、CE phase range、phase similarity drift、concentric gyro variation 與 phase movement rate。CE ratio drift 與單純向心時間變長不是最強訊號，因此研究表述要避免寫成 IMU 直接量肌肉疲勞。

## 2026-05-16：023 CE Phase-Aware Fatigue 與 RPE 驗證

日期：2026-05-16

狀態：implemented

目的：

用數據檢查「IMU 疲勞狀態是否能透過 CE phase 變化判斷」這個猜想。重點不是只看整段 rep 是否不穩，而是分開看 concentric / eccentric phase 的時間、gyro、PCA range、movement rate、waveform similarity。

實際改動：

- 新增 `tools/analyze_phase_aware_fatigue_rpe.py`；
- 新增 `docs/phase_aware_fatigue_ce_rpe_023_zh.md`；
- 使用 GT rep + GT CE phase segmentation；
- 合併第 018 版補上 yushuan 後的 Borg/RPE targets；
- 將訊號改成整段 session 先標準化，再切 phase，避免每一下 rep 重新 z-score 抹掉強度差異；
- 輸出 rep-level、set-level、by-exercise phase-aware fatigue correlation。

正式結果：

```text
output = artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/
rep rows = 1677
set rows = 143

Top set-level:
set_index_numeric                              raw Spearman =  0.4397
eccentric_pca_range_mean                       raw Spearman =  0.3377
eccentric_pca_range_last2                      raw Spearman =  0.3351
eccentric_wave_sim_to_first2_last_minus_first  raw Spearman = -0.3272
concentric_pca_range_mean                      raw Spearman =  0.3174
concentric_pca_range_last2                     raw Spearman =  0.3111
concentric_gyro_diff_rms_last2                 raw Spearman =  0.2830

Hypothesis checks:
phase_vector_similarity_slope                  raw Spearman = -0.2188
concentric_sec_last2_vs_first2                 raw Spearman =  0.1740
concentric_gyro_diff_rms_last2_vs_first2       raw Spearman =  0.1670
concentric_sec_slope                           raw Spearman =  0.1664
concentric_pca_movement_rate_last2_vs_first2   raw Spearman =  0.0091
```

結論：

數據支持「CE phase-aware fatigue」方向，但不支持「向心速度下降」作為唯一疲勞指標。比較有訊號的是 phase similarity 下降、phase PCA range 增加、concentric gyro 變化增加，以及部分動作的向心時間拉長。不同動作的最佳疲勞特徵差異很大，因此後續應使用 exercise-aware phase fatigue score。

下一步：

1. 設計每個動作各自的 phase fatigue feature set；
2. 做 set-level RPE prediction：progress + phase-aware fatigue + VO2 delayed load；
3. 用 few-shot subject calibration 校正個人 RPE 尺度；
4. 把 023 特徵接到 predicted CE segmentation，評估真實部署時的衰退。

## 2026-05-16：022 即時 RPE 特徵與 VO2 融合分析

日期：2026-05-16

狀態：implemented

目的：

使用補上 yushuan 後的 RPE set-level 特徵，合併 VO2 lag window，分析未來如果要即時估 RPE，IMU 波形和 VO2 應該抓哪些特徵。

背景問題：

使用者補上 yushuan RPE 後，進一步要求即時波形與 VO2 都納入考慮。VO2 不是動作同步訊號，而是有延遲的生理負荷，因此不能只看當下值，需要看 set 後 lag window。

實際改動：

- 新增 `tools/analyze_realtime_rpe_vo2_feature_correlations.py`；
- 新增 `docs/realtime_rpe_vo2_features_022_zh.md`；
- 合併：
  - `021_rpe_feature_correlation_with_yushuan/020_rpe_set_level_feature_dataset.csv`
  - `019_vo2_gt_waveform_relation/019_vo2_set_waveform_dataset.csv`
- 計算 VO2 lag `0/10/20/30/45/60s` 和 Borg/RPE 的 Spearman；
- 輸出即時 IMU + VO2 top feature 圖，以及 VO2-only lag correlation 圖。

正式結果：

```text
output = artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/
subjects = haoyu, yanz, yoru, yushuan
sets = 96
lag rows = 572

穩定 IMU / progress features:
set_index_numeric               raw Spearman ~=  0.41
movement_rate_cv                raw Spearman ~= -0.39 to -0.40
concentric_gain_last2_vs_first2 raw Spearman ~=  0.35 to  0.36
gyro_mag_diff_rms_slope         raw Spearman ~=  0.36
sim_to_first_slope              raw Spearman ~= -0.32
concentric_sec_slope            raw Spearman ~=  0.30 to  0.32

VO2 features:
vo2_mean_delta_subject_min @10s raw Spearman = -0.3500
vo2_mean_x_n_reps          @10s raw Spearman = -0.3158
vo2_peak_delta_subject_min @10s raw Spearman = -0.3108
vo2_slope                  @45s raw Spearman =  0.3639
```

結論：

即時 RPE 的核心應該是 IMU set-level fatigue state。VO2 有訊號，但目前更像延遲輔助負荷，且 raw VO2 方向不穩，容易受休息、動作種類、受試者基準與呼吸延遲影響。

下一步：

1. 將 VO2 改成 subject baseline-normalized VO2 delta / AUC；
2. 做 RPE prediction model：IMU fatigue state + VO2 delayed load + subject calibration；
3. 將模型設計成每個 rep 更新 IMU state，VO2 以 10-60 秒延遲更新；
4. 加入 few-shot subject calibration，降低不同人主觀 RPE 尺度差異。

## 2026-05-16：021 補上 yushuan 後的 RPE 特徵相關度

日期：2026-05-16

狀態：implemented

目的：

使用者補上 `yushuan0513workout.xlsx` 的完整 RPE 後，重新跑第 018 與第 020 版分析，確認 yushuan 是否能納入 RPE 訓練與相關度分析。

實際結果：

```text
018 rerun:
merged GT reps = 1677
trainable folders = haoyu, hsianshun, tsenyu, yanz, yoru, yushuan

021 output = artifacts_rep_classification/021_rpe_feature_correlation_with_yushuan/
rep-level rows = 1677
set-level rows = 143

Top rep-level:
rep_progress              raw Spearman =  0.5476
rep_index                 raw Spearman =  0.5256
set_index_numeric         raw Spearman =  0.5006
cumulative_tut_sec        raw Spearman =  0.4594
cumulative_eccentric_sec  raw Spearman =  0.4533
cumulative_concentric_sec raw Spearman =  0.4407

Top set-level:
set_index_numeric              raw Spearman =  0.4397
n_reps                         raw Spearman = -0.2927
pca_diff_rms_max               raw Spearman = -0.2873
pca_diff_rms_mean              raw Spearman = -0.2861
gyro_diff_gain_last2_vs_first2 raw Spearman =  0.1958
```

結論：

補上 yushuan 後，進度與累積 TUT 的訊號變得更強，支持使用 set-level fatigue trend 估 RPE。

## 2026-05-16：020 RPE 特徵相關度分析

日期：2026-05-16

狀態：implemented

目的：

先不訓練新模型，直接量化「哪些特徵和 Borg/RPE 有關」。這用來決定後續 RPE prediction 應該以 rep-level 瞬間波形、set-level 疲勞趨勢、TUT、還是 few-shot subject calibration 為主。

背景問題：

第 018 版顯示 GT 波形可以比 baseline 稍微改善 RPE regression，但還不夠強。使用者希望先把我認為有關的特徵用數據跑出相關度，而不是直接猜模型。

實際改動：

- 新增 `tools/analyze_rpe_feature_correlations.py`；
- 新增 `docs/rpe_feature_correlation_020_zh.md`；
- 使用 `018_borg_gt_waveform_relation_exclude_sparse/018_gt_rep_waveform_borg_dataset.csv`；
- 建立 rep-level 衍生特徵：
  - rep progress；
  - cumulative TUT；
  - cumulative concentric / eccentric duration；
  - velocity loss proxy；
  - waveform similarity decay；
  - last/first baseline change；
  - variability so far；
- 建立 set-level 衍生特徵：
  - final Borg/RPE；
  - total TUT；
  - duration / velocity / gyro `last2 vs first2`；
  - slope 與 variability；
- 輸出 raw、exercise-centered、subject-centered、subject+exercise-centered Spearman。

正式結果：

```text
output = artifacts_rep_classification/020_rpe_feature_correlation_analysis/
rep-level rows = 1396
set-level rows = 119

Top rep-level:
rep_progress              raw Spearman =  0.5166
set_index_numeric         raw Spearman =  0.5082
cumulative_tut_sec        raw Spearman =  0.4254
cumulative_eccentric_sec  raw Spearman =  0.4247
cumulative_concentric_sec raw Spearman =  0.4005
kg_x_rep                  raw Spearman =  0.3892

Top set-level:
set_index_numeric              raw Spearman =  0.4435
n_reps                         raw Spearman = -0.2700
pca_diff_rms_mean              raw Spearman = -0.2483
gyro_diff_gain_last2_vs_first2 raw Spearman =  0.2302
gyro_mag_diff_rms_slope        raw Spearman =  0.1878
```

結論：

RPE 最強訊號是進度與累積負荷：第幾組、第幾下、累積 TUT。單一下波形特徵只有弱相關；比較有用的是 set-level fatigue trend，尤其是 `last2 vs first2` 的 gyro 變化、duration 變長與 velocity loss。

下一步：

1. 做 set-level RPE model，而不是只做 rep-level regression；
2. 加入 per-subject few-shot calibration；
3. 補相對負重，例如 1RM 或個人 baseline；
4. 重新抽 raw-amplitude features，避免 rep 內 z-score 把絕對強度訊號消掉。

## 2026-05-16：018 Borg / REP 與 GT 波形特徵關聯上限測試

日期：2026-05-16

狀態：implemented

目的：

驗證在使用 ground-truth rep / phase 切割點的前提下，原始 IMU 波形、waveform similarity、TUT、向心 / 離心 duration 是否能預測 Borg/RPE。這是自動切割前的上限測試。

背景問題：

如果完美切割的原始波形都無法學到 Borg/RPE 關聯，繼續改善 rep segmentation 對「訓練建議」的價值就有限。使用者也補充：`X` 代表沒做完，空白代表沿用前一個 REP/Borg 值；`thomas0506workout` 沒有同名 workbook，不放入 training。

實際改動：

- 新增 `tools/analyze_borg_from_gt_waveform_features.py`；
- 新增 `docs/borg_waveform_relation_018_zh.md`；
- 在 `requirements.txt` 補上 `openpyxl`；
- 解析每位受試者同名 `.xlsx`：
  - `0..11` 欄位對應 rep index；
  - 空白 forward-fill；
  - `X` 排除；
  - `kg` / `KG` 作為重量特徵；
- 使用 GT rep / phase labels 抽取 TUT、phase ratio、9 軸統計、PCA waveform similarity；
- 以 subject-wise GroupKFold 評估 Borg/RPE regression。

正式結果：

```text
output = artifacts_rep_classification/018_borg_gt_waveform_relation/
raw Borg targets = 1425
completed Borg targets = 1416
merged GT waveform reps = 1408

含 yushuan sparse target:
global mean baseline MAE = 1.7274
exercise mean baseline MAE = 1.6489
waveform RF MAE = 1.6327
combined RF MAE = 1.6390

排除 yushuan sparse target:
output = artifacts_rep_classification/018_borg_gt_waveform_relation_exclude_sparse/
global mean baseline MAE = 1.7071
exercise mean baseline MAE = 1.6297
waveform RF MAE = 1.5554
combined RF MAE = 1.5741
```

結論：

GT 波形確實含有 Borg/RPE 訊號，但跨人預測不強。TUT-only 沒有比 metadata 好；waveform RF 比 global baseline 改善約 `0.15` Borg MAE，但仍無法達到高可信絕對 Borg 預測。下一步應改成 within-subject / few-shot calibration，或預測 set-level Borg slope / fatigue trend，而不是只預測單一下的絕對 Borg。

下一步：

1. 做 within-subject few-shot Borg calibration；
2. 改預測 `delta Borg`、最後一 rep Borg、set-level Borg slope；
3. 加入 set-level fatigue trend 特徵，例如 waveform similarity decline、rep duration slope、phase ratio slope；
4. 確認 GT 上限有效後，再比較 predicted segmentation 版本。

## 2026-05-16：017 向心 / 離心 Phase Split 診斷

日期：2026-05-16

狀態：implemented

目的：

拿目前最佳 rep boundary 方法 `DCP-DP-FS` 的輸出，檢查 rep 內部再切向心 / 離心的可行性，並輸出 phase IoU、phase TUT error 與上下兩排波形圖。

背景問題：

目前 count accuracy 和 rep boundary IoU 已經有比較表，但重訓建議真正需要的是 concentric / eccentric duration。若 phase split 不準，TUT 和 tempo 建議就不可靠。

實際改動：

- 新增 `tools/plot_phase_split_diagnostics.py`；
- 新增 `docs/phase_split_diagnostics_017_zh.md`；
- 從 `016_dense_candidate_dp_decoder/methods/dcp_dp_fs/rep_segments_manifest.csv` 讀取 predicted reps；
- 使用 `pca-reversal` 在每個 predicted rep 內切 phase；
- 輸出 overall / by-phase / by-exercise / by-subject phase IoU；
- 輸出 concentric / eccentric TUT error；
- 輸出 239 張上下兩排 phase waveform 圖。

正式結果：

```text
output = artifacts_rep_classification/017_phase_split_dcp_dp_fs/
true phase segments = 5364
predicted phase segments = 5308

Phase IoU@0.50 F1 = 0.5442
Phase IoU@0.75 F1 = 0.2103
Phase IoU@0.90 F1 = 0.0469

concentric IoU@0.50 F1 = 0.5683
eccentric  IoU@0.50 F1 = 0.5201

concentric TUT MAE = 2.3447 sec
eccentric  TUT MAE = 2.3454 sec
overall    TUT MAE = 2.3450 sec
```

結論：

目前 phase split 還不夠好。單純 `pca-reversal` 可在 `one_arm_db_row` 達到 `0.8271` 的 phase IoU@0.50 F1，但 `db_shoulder_press` 只有 `0.3027`，`db_rdl` 只有 `0.3868`。下一步應做 exercise-aware / subject-adaptive phase decoder，尤其要校正 phase ratio、split offset 和主要軸權重。

下一步：

1. 對 `db_shoulder_press`、`db_rdl` 看波形錯誤案例；
2. 加入 per-exercise phase feature：gyro zero-crossing、dominant-axis reversal、energy valley；
3. few-shot calibration 從 rep duration 擴充到 phase ratio / split offset；
4. 將 phase split 納入 SAPA-DP 創新方法。

## 2026-05-16：016 Dense Candidate Pool Dynamic Programming

日期：2026-05-16

狀態：implemented

目的：

根據第 015 版結論，候選點已經足夠多，但 final decoder 選錯。因此第 016 版改成從 dense candidate pool 做 clustering / pruning，再用 dynamic programming 一次選整組 rep boundaries。

背景問題：

014 的 LIFT-Fusion 仍依賴 uniform/autocorr prior 附近找 valley；015 顯示 `Fusion Candidate Pool` 在 `+/-20 samples` 的 recall 是 `0.8130`，但 `Final 014 LIFT-Fusion` 只有 `0.2745`。這表示需要 sequence-level decoding，而不是再增加局部 feature。

實際改動：

- 新增 `tools/evaluate_dense_candidate_dp_decoder.py`；
- 新增 `docs/dense_candidate_dp_decoder_016_zh.md`；
- 實作兩種方法：
  - `DCP-DP`：dense candidate pool + dynamic programming，不使用 few-shot；
  - `DCP-DP-FS`：加入 subject / exercise 前 3 下 labeled reps 的 duration calibration；
- 輸出 method comparison、by-exercise、by-subject、phase IoU 和 239 張 waveform 圖；
- 更新 `README.md` 與 `artifacts_rep_classification/RESULTS_INDEX.md`。

正式結果：

```text
output = artifacts_rep_classification/016_dense_candidate_dp_decoder/
active blocks = 239
true reps = 2720
waveform plots = 239

DCP-DP:
count exact = 0.6568
count +/-1 = 0.8856
count MAE = 1.4534 reps
rep IoU@0.50 F1 = 0.7485
rep IoU@0.75 F1 = 0.4490
rep IoU@0.90 F1 = 0.2010
phase IoU@0.50 F1 = 0.5202

DCP-DP-FS:
count exact = 0.6780
count +/-1 = 0.9280
count MAE = 0.6271 reps
rep IoU@0.50 F1 = 0.7834
rep IoU@0.75 F1 = 0.4648
rep IoU@0.90 F1 = 0.2051
phase IoU@0.50 F1 = 0.5442
```

和 014 比較：

```text
014 LIFT-Fusion rep IoU@0.50 / 0.75 / 0.90 = 0.7855 / 0.4610 / 0.1920
014 MAXXYT-MAP   rep IoU@0.50 / 0.75 / 0.90 = 0.7648 / 0.4380 / 0.2017
016 DCP-DP-FS    rep IoU@0.50 / 0.75 / 0.90 = 0.7834 / 0.4648 / 0.2051
```

結論：

`DCP-DP-FS` 是目前高 IoU 和 phase split 最好的方法，但改善幅度仍有限。這證明 DP 方向有效，但只靠 duration calibration 不夠；下一步要學 subject-specific boundary offset、axis weights 和 exercise-specific score。

下一步：

1. 對 few-shot calibration 加入 boundary offset，而不是只用 duration；
2. 對 `db_shoulder_press`、`db_bench_press` 做錯誤圖分析；
3. 訓練 candidate-level boundary scorer，再交給 DP 做 sequence decoding；
4. 保留 015 的 candidate recall 指標，避免只看 final IoU。

## 2026-05-16：015 Boundary Candidate Recall 分析

日期：2026-05-16

狀態：implemented

目的：

驗證目前 rep boundary 不準的真正原因。第 014 版已經顯示 count-level 接近可用，但 IoU@0.90 仍低；這次不先改模型，而是量化每個 GT internal boundary 附近是否存在候選切點。

背景問題：

如果 GT boundary 附近根本沒有候選點，代表要找新特徵；如果候選點有在附近但最後方法沒選到，代表瓶頸在 scoring / decoding / personalization。

實際改動：

- 新增 `tools/analyze_boundary_candidate_recall.py`；
- 新增 `docs/boundary_candidate_recall_015_zh.md`；
- 輸出候選池與 final boundary 的 `+/-5`、`+/-10`、`+/-20`、`+/-50 samples` recall；
- 同時輸出 overall、by-exercise、by-subject CSV 與結果圖；
- 更新 `artifacts_rep_classification/RESULTS_INDEX.md`。

正式結果：

```text
output = artifacts_rep_classification/015_boundary_candidate_recall_analysis/
active blocks = 239
true internal boundaries = 2481

Raw Gyro + Energy Valleys:
mean candidates per block = 306.61
median nearest error = 4 samples
recall +/-20 samples = 0.9964

Fusion Candidate Pool:
mean candidates per block = 174.45
median nearest error = 6 samples
recall +/-20 samples = 0.8130

Final 014 LIFT-Fusion:
mean boundaries per block = 10.57
median nearest error = 54 samples
recall +/-20 samples = 0.2745

Final 014 MAXXYT-MAP:
mean boundaries per block = 10.58
median nearest error = 57 samples
recall +/-20 samples = 0.2652
```

結論：

目前問題不是「完全沒有特徵」。dense candidate pool 幾乎都能在 GT 附近找到候選，但 final decoder 只選到少數正確候選。也就是：

```text
候選點足夠多
-> 候選太密
-> uniform/autocorr prior 太粗
-> scoring / sequence decoding 選錯
-> IoU@0.90 很低
```

下一步：

1. 從 dense candidate pool 做 candidate pruning；
2. 用 dynamic programming / sequence decoding 在候選點之間選整組 boundary；
3. 加入 subject calibration，學個人 duration scale、boundary offset、axis weight；
4. 優先處理 `db_shoulder_press`、`db_bench_press`、`yanz0510workout`、`yoru0511workout`。

## 2026-05-16：014 文獻啟發方法比較與 LIFT-Fusion

日期：2026-05-16

狀態：implemented

目的：

根據前面討論的 StayFit、Maxxyt、M-Fitness、CaRaCount / DTW 等 repetition counting / waveform segmentation 論文，實作可在目前 IMU active-only 資料上比較的 rep boundary 方法，並提出一個融合式新方法 `LIFT-Fusion`。

背景問題：

010 / 011 已能達到約 `0.73` 的 IoU@0.50 F1，但高精度 IoU@0.90 仍只有約 `0.16`。單純 DS-MS-TCN seq-to-seq baseline 也沒有超過 classical boundary 方法。需要把文獻中的 count consensus、energy segmentation、template personalization 拿來做可量化比較。

假設：

- 這版只評估已在運動中的 active set，不處理休息或 set detection；
- 每個 active block 保留 ground-truth exercise hint，目標是先把 rep boundary 做準；
- `CARA-DTW-FS` 和 `LIFT-Fusion` 使用少量同 subject/exercise 標註 rep 作為 personalization template；
- 預設 template refinement 使用快速 resampled shape distance，若要完整 DTW 可加 `--use-dtw-shape-cost`，但計算時間較高。

實際改動：

- 新增 `tools/evaluate_literature_inspired_rep_methods.py`；
- 新增五個方法：
  - `STAYFIT-BA`：best-axis periodic peak cutting；
  - `MAXXYT-MAP`：multi-axis adaptive peak aggregation + gyro valley refinement；
  - `MFIT-FSTE`：frequency-weighted short-time energy valley segmentation；
  - `CARA-DTW-FS`：few-shot template alignment；
  - `LIFT-Fusion`：PCA/autocorr + multi-axis count consensus + energy/gyro valley + few-shot template；
- 輸出總比較 CSV / PNG、每動作 heatmap、每人 heatmap、每方法 metrics、239 張 waveform all-set plots；
- 新增 `docs/literature_inspired_rep_methods_014_zh.md`；
- 更新 `artifacts_rep_classification/RESULTS_INDEX.md`。

正式結果：

```text
output = artifacts_rep_classification/014_literature_inspired_rep_methods/
sessions = 14
active blocks = 239
true sets = 236
true reps = 2720
waveform plots = 239

LIFT-Fusion:
count exact = 0.6737
count +/-1 = 0.9280
count MAE = 0.6314 reps
rep IoU@0.50 F1 = 0.7855
rep IoU@0.75 F1 = 0.4610
rep IoU@0.90 F1 = 0.1920
phase IoU@0.50 F1 = 0.5319

MAXXYT-MAP:
count exact = 0.6822
count +/-1 = 0.9364
count MAE = 0.6186 reps
rep IoU@0.50 F1 = 0.7648
rep IoU@0.75 F1 = 0.4380
rep IoU@0.90 F1 = 0.2017
phase IoU@0.50 F1 = 0.5305
```

結論：

第 014 版確實比 010 / 011 改善：

```text
010 UGV  IoU@0.50 / 0.75 / 0.90 = 0.7278 / 0.3949 / 0.1626
011 MFBS IoU@0.50 / 0.75 / 0.90 = 0.7382 / 0.4106 / 0.1621
014 LIFT IoU@0.50 / 0.75 / 0.90 = 0.7855 / 0.4610 / 0.1920
```

但仍未達到 IoU@0.90 F1 90% 的目標。現在可以說 count-level 已接近可用，因為 LIFT `count +/-1 = 0.9280`，MAXXYT-MAP `count +/-1 = 0.9364`；但 boundary-level 還不足以支撐高精度 TUT 或向心/離心切割。

下一步：

優先做 boundary candidate recall 分析，而不是直接再加模型：

1. 檢查每種方法在 GT boundary ±5 / ±10 / ±20 samples 是否有候選點；
2. 針對 `db_bench_press`、`db_shoulder_press`、`yoru0511workout`、`yanz0510workout` 做錯誤波形分析；
3. 把 few-shot calibration 從「用 template 修 boundary」升級成「學個人 duration scale、axis weight、boundary offset」；
4. 再重新評估 IoU@0.90、boundary median error、within-10-sample rate。

## 2026-05-16：013 方法比較表加入正式方法名稱

日期：2026-05-16

狀態：implemented

目的：

讓 count / IoU / TUT 比較表不只顯示 010/011/012 實驗代號，而是每個方法都有可放入報告的方法名稱。

背景問題：

原本 `013_count_iou_tut_method_table` 的第一欄是 artifact 代號，例如 `010_universal_gyro_valley`。這方便追蹤檔案，但不適合直接放入論文式比較表。

假設：

- artifact 代號仍需保留，方便回查原始輸出；
- 圖表應使用短方法名，避免 x 軸過長；
- CSV 應同時包含正式名稱、artifact id 與方法說明。

實際結果：

- 更新 `tools/build_count_iou_tut_table.py`；
- CSV 新增 `method_name`、`method_id`、`method_description`；
- 圖表改用 `method_name`；
- `summary.json` 新增 method legend；
- 重跑 `artifacts_rep_classification/013_count_iou_tut_method_table/`。

下一步：

後續報告使用 `method_name` 作為主名稱，`method_id` 只作為附註或實驗追蹤。

## 2026-05-16：012 DS-MS-TCN 9 軸 sequence baseline

日期：2026-05-16

狀態：implemented

目的：

參考 DS-MS-TCN 論文的資料呈現方式，新增可訓練的 seq-to-seq baseline，讓本專案可以用 sample-wise F1、segment IoU F1、rep boundary IoU、phase IoU、confusion matrix 與 waveform 切割圖比較方法差異。

背景問題：

010/011 已能用 classical / supervised boundary scoring 做 rep IoU 比較，但沒有 sample-wise temporal segmentation baseline，也無法直接和 DS-MS-TCN 這類方法的呈現方式對齊。

假設：

- 先實作 adapted DS-MS-TCN，而不是完全複製 Otago task；
- 使用者已決定 012 使用 9 軸輸入；
- DS-MS-TCN / MS-TCN 之間比較 sample-wise F1 與 macro segment IoU；
- DS-MS-TCN 與 010/011 之間只比較 rep boundary IoU，避免 metric 不公平。

預計改動：

- 新增 9 軸 DS-MS-TCN / MS-TCN model；
- 新增 training/evaluation script；
- 新增 012 vs 010/011 comparison script；
- 新增中文方法文件與結果索引；
- 先跑 smoke test，不把 smoke test 當正式準確率。

不做的事：

- 不先加入 CNN、Transformer、CNN-LSTM baseline；
- 不宣稱 1 epoch smoke test 的結果代表正式模型效能；
- 不把 9 軸 DS-MS-TCN 與 6 軸 classical 方法當成同輸入公平比較。

預期改善：

- 能直接跑出論文式 method comparison table / bar chart；
- 能檢查 micro label 對 macro segmentation 是否有幫助；
- 能把 sample-wise prediction 轉成 rep / phase segment 再用 IoU 評估。

評估指標：

- sample-wise macro F1；
- sample-wise micro F1；
- macro segment IoU F1@0.50 / 0.75 / 0.90；
- rep boundary IoU F1@0.50 / 0.75 / 0.90；
- phase split IoU F1@0.50 / 0.75 / 0.90；
- 每人、每動作 rep IoU 圖；
- macro/micro confusion matrix；
- waveform boundary examples。

風險：

- 完整 5-fold 訓練耗時高於 classical methods；
- full-session `other` 類容易壓過 active samples；
- MS-TCN 沒有 micro labels，轉 rep boundary 時只能使用 macro segment proxy，rep IoU 解讀需保守。

審核結果：

使用者要求直接實作計畫。

實際結果：

- 新增 `models/ds_ms_tcn.py`；
- 新增 `tools/train_ds_ms_tcn_9axis.py`；
- 新增 `tools/compare_ds_ms_tcn_9axis.py`；
- 新增 `docs/ds_ms_tcn_9axis_comparison_zh.md`；
- 更新 `README.md` 與 `artifacts_rep_classification/RESULTS_INDEX.md`；
- `python -m py_compile` 通過；
- exercise-only smoke test 通過；
- full-session + other smoke test 通過；
- smoke comparison 圖表輸出成功。
- 正式 5-fold exercise-only 已完成：DS-MS-TCN `Rep F1@0.50 = 0.4765`、`Rep F1@0.90 = 0.1301`；
- 正式 5-fold full-session + other 已完成：DS-MS-TCN `Rep F1@0.50 = 0.3590`、`Rep F1@0.90 = 0.0759`；
- 012 正式比較圖已輸出到 `artifacts_rep_classification/012_ds_ms_tcn_9axis_method_comparison/`。

下一步：

目前 012 沒有超過 010/011。下一步若要提升，應優先改 rep boundary decoding，而不是只加訓練 epoch：

```bash
sample-wise macro prediction
-> phase-aware smoothing
-> per-exercise duration prior
-> rep boundary decoding
-> IoU@0.90 objective / calibration
```

## 2026-05-15：建立研究與實驗規劃文件

日期：2026-05-15

狀態：implemented

目的：

建立後續方法改善的共同規劃基準，避免直接實作造成方向發散或重跑成本過高。

背景問題：

目前 rep segmentation 已有多個 baseline，但仍有 over-segmentation、phase split 尚未穩定、TUT 無法正式作為訓練建議等問題。使用者希望後續每次實作前先提出計畫，審核後再執行。

假設：

- 第一階段應只整理文件，不動模型與 artifacts；
- 目前最值得優先比較的方法是 PCA、autocorrelation、wavelet、DTW、state machine 與 tiny TCN；
- Luckfox Pico Zero 適合輕量即時推論，不適合第一階段就導入 Transformer。

預計改動：

- 新增 `docs/experiment_plan_zh.md`；
- 新增 `docs/change_log_zh.md`；
- 在 README 補上文件連結。

不做的事：

- 不修改 rep segmentation 程式；
- 不重新產生圖表；
- 不重跑完整 pipeline；
- 不導入新套件。

預期改善：

- 後續每次變更都有明確目的與審核流程；
- 方法比較與文獻引用可追溯；
- 可以更清楚判斷哪些方法適合即時端，哪些方法適合 post-set 分析。

評估指標：

- 文件是否清楚列出方法、優先度、指標、風險；
- README 是否能引導到新文件；
- git diff 是否只包含文件。

風險：

- 文件太長會降低維護意願；
- 若後續實驗結果改變，規劃文件需要同步更新。

審核結果：

使用者回覆「好」，同意先做第一階段文件整理。

實際結果：

已新增實驗規劃文件與變更紀錄文件，並更新 README 文件連結。

下一步：

等待使用者審核文件內容。若同意，下一個 proposal 建議是「PCA + autocorrelation-constrained peak segmentation」。

## 2026-05-15：PCA + Autocorrelation + Peak

日期：2026-05-15

狀態：implemented

目的：

降低目前 PCA peak 類方法的 over-segmentation，讓 predicted reps 數量更接近 true reps，同時保留 boundary-level IoU 評估。

背景問題：

目前 `pca-extrema` 有較高 recall，但 predicted reps 明顯多於 true reps。`pca-extrema-fft` 可降低過切，但 FFT 是整段頻率估計，對 local boundary 的解釋力有限。

假設：

同一個 set 內 repetition 具有局部週期性。用 autocorrelation 估計 dominant period 後，將它轉成 peak distance constraint，可以減少雜訊峰造成的過切。

預計改動：

- 在 `tools/evaluate_rep_segmentation_classification.py` 新增 segment method：
  - `pca-autocorr`
- 新增 autocorrelation period estimation helper；
- 用 period 估計限制 `find_peaks` 的 distance；
- 輸出與既有方法相同的 IoU、confusion matrix、waveform plot。

不做的事：

- 不做 Wavelet；
- 不做 DTW；
- 不做 phase split；
- 不做 embedded deployment。

預期改善：

- prediction ratio 降低；
- IoU@0.50 precision 提升；
- IoU@0.50 F1 提升；
- waveform 圖中過密切割線減少。

評估指標：

- overall IoU@0.50 F1；
- per-exercise IoU@0.50 F1；
- predicted / true reps ratio；
- set-level count MAE；
- 210 組 set waveform 圖。

風險：

- 若某些動作節奏變化大，autocorrelation 估計的 period 可能失準；
- 若 set 內混入 rest 或拿放啞鈴，週期估計會被污染；
- 需要先處理 active set detection 才能更穩。

審核結果：

使用者回覆「可以開始做」，同意實作此 proposal。

實際結果：

已新增 `pca-autocorr` segmentation method，並更新一鍵 pipeline、README、實驗規劃與文獻比較文件。

實作重點：

- 在 PCA principal motion signal 上估計 autocorrelation dominant period；
- 用 dominant period 限制 peak distance；
- 用 peak / trough midpoint 形成 rep boundary；
- 自相關實作採用 FFT-based autocorrelation，避免直接 `np.correlate` 在長 set 上造成 O(n^2) 計算瓶頸。

數值結果：

```text
true reps: 2424
predicted reps: 4846
classified reps: 1627
IoU@0.25 F1: 0.4116
IoU@0.50 precision: 0.2070
IoU@0.50 recall: 0.4138
IoU@0.50 F1: 0.2759
IoU@0.75 F1: 0.0930
classification accuracy: 0.8482
macro F1: 0.8240
weighted F1: 0.8490
```

與 `pca-extrema-fft` 相比：

```text
predicted reps: 5942 -> 4846
IoU@0.50 F1: 0.2044 -> 0.2759
IoU@0.75 F1: 0.0304 -> 0.0930
set-level mean best IoU: 0.3479 -> 0.3606
set-level prediction ratio: 0.9211 -> 0.9774
```

輸出結果：

- `artifacts_rep_classification/pca_autocorr_8class_5fold/`
- `artifacts_rep_classification/methods_comparison/`
- `artifacts_rep_classification/waveform_method_comparison/sets_all/`
- `artifacts_rep_classification/waveform_method_comparison/set_level_results/`

下一步：

下一個建議 proposal 是「active set detection + exercise-aware prior」。原因是 `pca-autocorr` 已把 prediction ratio 拉近到 `0.9774`，但 IoU@0.50 false positives 仍有 `3843`；如果能先把不穩定的 set/rest 邊界處理好，再依動作類別限制合理 rep duration，應該比直接加更複雜模型更務實。

## 下一個建議 proposal：Active Set Detection + Exercise-aware Prior

日期：待定

狀態：proposal

目的：

在進一步做 Wavelet、DTW 或 TCN 前，先降低 set/rest 邊界污染與跨動作週期差異造成的錯切。

背景問題：

目前 `pca-autocorr` 已經改善過切，但仍有不少 false positives。主要可能來自：

- set 起訖包含拿放啞鈴或姿勢調整；
- 不同動作合理 rep duration 不同；
- peak distance 仍是全域比例，沒有使用 exercise-specific prior。

假設：

先做 active set trimming，再根據 exercise 設定 period / duration prior，可以比直接加入更複雜模型更有效降低 false positives。

預計改動：

- 從 labeled set block 內部再用 motion energy trim 開頭與結尾；
- 從 ground truth training folds 估計 per-exercise rep duration quantile；
- 在 validation / test block 中用該 exercise prior 限制 autocorr period；
- 新增 method 名稱，例如 `pca-autocorr-prior`。

不做的事：

- 不做 Wavelet；
- 不做 DTW；
- 不做 phase split；
- 不做 Luckfox 部署。

預期改善：

- IoU@0.50 precision 提升；
- false positives 降低；
- boundary start/end error 降低；
- set-level prediction ratio 維持接近 1。

評估指標：

- IoU@0.50 F1；
- false positives；
- per-exercise IoU@0.50 F1；
- set-level count MAE；
- waveform boundary comparison。

風險：

- 若 prior 從全部資料估計，會有資料洩漏疑慮；正式比較時應改成 fold-wise prior；
- 若 trim 太 aggressive，可能漏掉第一下或最後一下 rep。

審核結果：

待使用者確認。

實際結果：

待實作。

下一步：

等待使用者審核。

## 2026-05-15：Active / Set Detection Baseline Verification

日期：2026-05-15

狀態：implemented

目的：

先驗證目前是否能從 IMU waveform 切出「有沒有運動 / set 區間」，避免後續 rep segmentation 被 rest、拿放啞鈴、準備姿勢污染。

背景問題：

目前 `set_blocks_from_labels()` 是使用 `action_type != rest` 的標註產生 set block，這不是模型預測出來的 active detection。因此目前還沒有 active/set detection 的 precision、recall、IoU、F1。

假設：

- 若 active/set detection 已經很差，rep boundary 不可能接近 90%；
- `action_type != big_rest/rest/none` 可作為寬鬆 set-level active ground truth；
- `phase in {concentric, eccentric}` 可作為嚴格 rep-active ground truth；
- 先用 IMU energy / variance baseline 就能看出 active detection 是否是主要瓶頸。

預計改動：

- 新增 `tools/evaluate_active_set_detection.py`；
- 支援兩種 ground truth：
  - `action`：`action_type` 非 rest；
  - `phase`：`phase` 為 concentric/eccentric；
- 支援 baseline：
  - `oracle-action`；
  - `imu-energy`；
  - `imu-variance`；
- 輸出 active detection 的 CSV 指標與 timeline 圖。

不做的事：

- 不修改既有 rep segmentation 方法；
- 不重跑所有 rep 方法；
- 不加入 supervised model；
- 不做 Luckfox 部署。

預期改善：

- 釐清目前 rep 錯誤是否來自前段 active/set detection；
- 建立 active detection 指標，後續可和 rep IoU 串起來分析；
- 找出 set 邊界、rest、拿放啞鈴造成的主要錯誤。

評估指標：

- sample-level precision / recall / F1 / accuracy；
- segment-level IoU@0.50 precision / recall / F1；
- false active duration；
- missed active duration；
- start/end boundary error；
- per-subject、per-exercise 分析。

風險：

- left-side files 目前多為 `phase=none` 且 action 固定，可能不適合用來評估 active detection；
- energy threshold 可能需要用 training subjects 估計，否則會有資料洩漏問題；
- action-level ground truth 和 phase-level ground truth 的定義不同，結果不能混在一起解讀。

審核結果：

使用者回覆「好」，同意開始。

實際結果：

已新增 `tools/evaluate_active_set_detection.py`，並輸出 active / set detection 的 CSV 與圖表。

輸出結果：

- `artifacts_active_detection/active_detection_metrics.csv`
- `artifacts_active_detection/active_detection_metrics_by_subject.csv`
- `artifacts_active_detection/active_detection_metrics_by_exercise.csv`
- `artifacts_active_detection/active_detection_overall_f1.png`
- `artifacts_active_detection/active_detection_sample_f1_by_subject.png`
- `artifacts_active_detection/active_detection_segment_f1_by_subject.png`
- `artifacts_active_detection/timeline_examples/`

數值結果：

```text
target=action, method=oracle-action:
sample F1 = 1.0000
segment IoU@0.50 F1 = 1.0000
true segments = 68
predicted segments = 68

target=action, method=imu-energy:
sample precision = 0.7378
sample recall = 0.3036
sample F1 = 0.4301
segment IoU@0.50 F1 = 0.0000
true segments = 68
predicted segments = 3956

target=action, method=imu-variance:
sample precision = 0.7332
sample recall = 0.3008
sample F1 = 0.4266
segment IoU@0.50 F1 = 0.0000
true segments = 68
predicted segments = 4275

target=phase, method=imu-energy:
sample precision = 0.2504
sample recall = 0.3547
sample F1 = 0.2936
segment IoU@0.50 F1 = 0.0346
true segments = 213
predicted segments = 2731
matched segments = 51
mean matched IoU = 0.8273
mean start error = 1.7233 sec
mean end error = 4.0872 sec

target=phase, method=imu-variance:
sample precision = 0.2251
sample recall = 0.3187
sample F1 = 0.2638
segment IoU@0.50 F1 = 0.0149
true segments = 213
predicted segments = 3004
matched segments = 24
mean matched IoU = 0.7115
mean start error = 3.5000 sec
mean end error = 5.0607 sec
```

結論：

目前資料確實有可用的 active/set 標註，但 naive IMU energy / variance detector 只能抓到局部高能量片段，無法穩定形成完整 set。這會讓後續 rep segmentation 被過碎的候選區段污染，因此 active/set detection 是下一個必須優先改善的瓶頸。

下一步：

建議下一個 proposal 改成「Hysteresis Active Set Detector + Exercise-aware Duration Prior」：

- 低通 / moving average 後做雙閾值 hysteresis；
- 對短 gap 合併、對不合理短片段刪除；
- 用 training subjects 估每個動作的 set duration / rep duration prior；
- 以 subject-wise split 驗證，避免驗證人的資料進入 threshold 或 prior 估計。

## 2026-05-15：Active Detection Method Pruning + Window RF

日期：2026-05-15

狀態：implemented

目的：

停止把時間花在已證明 segment-level 正確率很低的 `imu-energy` / `imu-variance` threshold baseline，改試更有機會泛化到新人的 active / set detector。

背景問題：

`imu-energy` / `imu-variance` 能抓到局部高能量片段，但會把完整 set 切成大量碎片。這類方法即使繼續微調 threshold，也不太可能直接解決 set boundary。

假設：

- 若 active/rest window-level 特徵能被 subject-wise 模型學到，代表問題不是 IMU 訊號本身不可分；
- 若 window-level F1 高但 segment IoU 低，主要瓶頸會轉向 set boundary post-processing 或 ground truth 定義；
- 低分 baseline 保留為歷史對照，不放在預設實驗流程。

預計改動：

- `tools/evaluate_active_set_detection.py` 預設方法改成 `oracle-action` + `imu-hysteresis`；
- 新增 `imu-hysteresis`：motion envelope、雙閾值 hysteresis、長 gap merge、最短 set duration filter；
- 新增 `tools/evaluate_active_set_window_classifier.py`；
- 使用 subject-wise GroupKFold 訓練 `window-rf` active/rest classifier；
- 輸出 window confusion matrix、F1 圖、每個驗證人的 timeline 切割線。

不做的事：

- 不再預設重跑 `imu-energy` / `imu-variance`；
- 不把 threshold-tuning 的 exploratory 輸出列為正式結果；
- 不修改 rep segmentation 主流程。

預期改善：

- 快速判斷 active/rest 是否可學；
- 減少低分 baseline 重跑成本；
- 找出「window active classification」與「set boundary segmentation」哪一段是瓶頸。

評估指標：

- window / sample-level F1；
- segment-level IoU@0.50 F1；
- predicted segments vs. true segments；
- window confusion matrix；
- subject-wise fold manifest。

風險：

- `action_type` 標註比較像整組 set 的動作類別，不一定等同「手正在動」；
- 若 action block 內含準備姿勢或短暫停頓，active/rest classifier 可能被 segment IoU 懲罰；
- window classifier 需要再接更好的 set boundary post-processing。

審核結果：

使用者要求「不要花時間跑那些正確率很低的方法，但可以嘗試別的方法」，因此本次直接執行方法剪枝與新方法試驗。

實際結果：

`imu-hysteresis` tuned action result：

```text
sample precision = 0.7299
sample recall = 0.5630
sample F1 = 0.6357
segment IoU@0.50 F1 = 0.1429
true segments = 68
predicted segments = 114
matched segments = 13
```

`window-rf` subject-wise 5-fold action result：

```text
num windows = 51420
positive windows = 39323
sample precision = 0.7673
sample recall = 0.9923
sample F1 = 0.8654
segment IoU@0.50 F1 = 0.1978
true segments = 68
predicted segments = 23
matched segments = 9
mean matched IoU = 0.8170
```

輸出結果：

- `artifacts_active_detection/001_window_rf_action_5fold/active_detection_metrics.csv`
- `artifacts_active_detection/001_window_rf_action_5fold/active_detection_metrics_by_subject.csv`
- `artifacts_active_detection/001_window_rf_action_5fold/fold_manifest.csv`
- `artifacts_active_detection/001_window_rf_action_5fold/window_confusion_matrix.png`
- `artifacts_active_detection/001_window_rf_action_5fold/window_rf_active_detection_f1.png`
- `artifacts_active_detection/001_window_rf_action_5fold/timeline_examples/`

結論：

`window-rf` 證明 active/rest 在 window-level 是可學的，sample F1 已明顯高於 threshold baseline。但 segment IoU 仍低，原因不是單純分類器不會分，而是 action-level ground truth 與實際「手正在動」不完全一致，加上 post-processing 會把多組 set 黏在一起或切掉 set 開頭。下一步不應再跑低分 energy baseline，而應改善 set boundary 定義與後處理。

下一步：

建議下一個 proposal 是「Boundary-aware Active Set Detector」：

- 用 window classifier 輸出 active probability；
- 在 probability 上做 valley-based split，避免多組 set 被黏成一段；
- 用 `set` / `rep` / `phase` 標註建立更乾淨的 set boundary target；
- 分別回報 `movement-active` 與 `set-action` 兩種 IoU，不再混在同一個指標解讀。

## 2026-05-15：Result Folder Versioning

日期：2026-05-15

狀態：implemented

目的：

讓每次正式輸出的結果資料夾前面都有版本編號，避免不知道哪一版實驗產生哪一批圖與數值。

背景問題：

使用者指出結果資料夾需要標號，才能知道這是哪一版方法改出來的結果。

假設：

- 已經正式保留的 active detection 結果先標為 `001`；
- 後續每個正式實驗依序使用 `002_...`、`003_...`；
- exploratory / 低分測參數輸出不列入正式版本。

預計改動：

- 將 `artifacts_active_detection/window_rf_action_5fold/` 改名為 `artifacts_active_detection/001_window_rf_action_5fold/`；
- 新增 `artifacts_active_detection/RESULTS_INDEX.md`；
- 更新 README 與工具預設 output dir。

不做的事：

- 不重跑模型；
- 不改數值結果；
- 不保留先前低分 exploratory 輸出。

預期改善：

- 可直接從資料夾名稱知道版本；
- 文件與 artifact 路徑一致；
- 後續比較不同方法時不會覆蓋前一版正式結果。

評估指標：

- `find artifacts_active_detection -maxdepth 2 -type d` 能看到版本化資料夾；
- README、change log、summary path 不再指向舊資料夾；
- git diff 只包含 rename、文件與 index。

風險：

- 如果外部腳本硬編碼舊路徑，需要同步改成新路徑。

審核結果：

使用者要求結果資料夾前面標號，因此直接執行。

實際結果：

正式結果資料夾：

- `artifacts_active_detection/001_window_rf_action_5fold/`

版本索引：

- `artifacts_active_detection/RESULTS_INDEX.md`

下一步：

下一個正式結果建議使用 `002_boundary_aware_active_set/`，用來放 valley split / boundary-aware active set detector 的結果。

## 2026-05-15：Active-only Rep / Classification / Phase Evaluation

日期：2026-05-15

狀態：implemented

目的：

先不要處理休息資料，也先拔掉「有沒有在運動」這個第一步。直接使用標註中的運動區段，確認當資料已經在運動中時，rep 切割、動作分類、向心/離心切割能達到什麼程度。

背景問題：

前一版 active/set detection 結果顯示，active/rest window-level 可以學到，但 set-level boundary 很差。若繼續把休息、準備姿勢、拿放啞鈴一起丟進 rep segmentation，會無法判斷問題到底來自 active detection 還是 rep boundary 本身。

假設：

- `phase in {concentric, eccentric}` 可代表已經在運動中的資料；
- 用每組 set 的 active phase span 作為候選 block，可以移除大量 rest / preparation contamination；
- 若 active-only rep IoU 明顯提升，表示前段 active/set detection 是主要瓶頸之一；
- phase split 可以先用 PCA reversal baseline，並用 IoU 呈現。

預計改動：

- 在 `tools/evaluate_rep_segmentation_classification.py` 新增 `--block-source active-phase-span`；
- 新增 phase split evaluation：
  - true phase：資料內 `phase` 標註；
  - predicted phase：每個 predicted rep 內用 PCA reversal split；
  - 指標：IoU@0.25 / 0.50 / 0.75 precision、recall、F1；
- 輸出 `phase_split_metrics.csv`、`phase_split_metrics_by_phase.csv` 與結果圖；
- 新增 `artifacts_rep_classification/RESULTS_INDEX.md`；
- 只跑 active-only `labels` 與 `pca-autocorr`，不跑低分 active detection 或 threshold baseline。

不做的事：

- 不使用休息資料；
- 不做 active/rest detector；
- 不重跑 `dominant-axis`、`short-time-energy` 等低分方法；
- 不把 phase split 做成 supervised model。

預期改善：

- rep segmentation IoU 應顯著高於舊的 action-block 結果；
- 能判斷目前分類模型在已切好 rep 或近似 rep 上的實際能力；
- 能初步量化向心/離心切割是否可用。

評估指標：

- rep IoU@0.25 / 0.50 / 0.75 F1；
- subject-wise 5-fold exercise classification accuracy / macro F1；
- phase IoU@0.25 / 0.50 / 0.75 F1；
- confusion matrix；
- per-exercise rep IoU heatmap；
- per-phase split IoU heatmap。

風險：

- `active-phase-span` 使用標註來移除 rest，因此這不是完整即時系統，只是分離問題來源的實驗；
- phase split 的 PCA reversal 仍是 rule-based baseline，會受到 rep boundary error 放大；
- action classification 的訓練樣本來自 predicted reps，若 predicted rep 與真實 rep 對齊不好，分類結果會被 label matching 影響。

審核結果：

使用者要求「只先處理有在運動的部分，先不要使用休息時候的資料」，因此直接實作並跑正式結果。

實際結果：

`001_active_only_labels_8class_5fold`：

```text
segment method = labels
block source = active-phase-span
true reps = 2424
predicted reps = 2424
classified reps = 2424
rep IoU@0.50 F1 = 1.0000
exercise classification accuracy = 0.8197
exercise macro F1 = 0.8198
phase split method = pca-reversal
phase IoU@0.50 F1 = 0.8333
```

`002_active_only_pca_autocorr_8class_5fold`：

```text
segment method = pca-autocorr
block source = active-phase-span
true reps = 2424
predicted reps = 2328
classified reps = 2290
rep IoU@0.25 F1 = 0.9247
rep IoU@0.50 F1 = 0.7083
rep IoU@0.75 F1 = 0.3308
exercise classification accuracy = 0.8459
exercise macro F1 = 0.8457
phase split method = pca-reversal
phase IoU@0.25 F1 = 0.6860
phase IoU@0.50 F1 = 0.4063
phase IoU@0.75 F1 = 0.1671
```

與舊 `pca-autocorr` action-block 結果相比：

```text
rep IoU@0.50 F1: 0.2759 -> 0.7083
predicted reps: 4846 -> 2328
true reps: 2424
```

結論：

只處理已在運動中的區段後，rep segmentation 明顯變好，證明 rest / preparation contamination 是主要問題之一。動作分類在 active-only predicted reps 上約 `0.8459`，已經比 oracle labels 的 `0.8197` 略高，可能是因為 predicted reps 中只保留 IoU 足夠可標註的樣本，較難的樣本被排除。phase split 在真實 rep 邊界下可達 IoU@0.50 F1 `0.8333`，但在 predicted reps 上掉到 `0.4063`，表示 phase split 目前主要受 rep boundary error 影響。

輸出結果：

- `artifacts_rep_classification/001_active_only_labels_8class_5fold/`
- `artifacts_rep_classification/002_active_only_pca_autocorr_8class_5fold/`
- `artifacts_rep_classification/RESULTS_INDEX.md`

下一步：

建議下一個正式實驗是 `003_active_only_boundary_refinement/`：

- 不碰休息資料；
- 只在 active-phase-span 內改善 rep boundary；
- 針對 IoU@0.75 低的問題做 boundary refinement；
- 比較 midpoint、PCA reversal、DTW template refinement 或 per-exercise duration prior。

## 2026-05-15：Active-only PCA Autocorr Boundary Refinement

日期：2026-05-15

狀態：implemented

目的：

在不使用休息資料的 active-only 條件下，改善 `pca-autocorr` 的 rep boundary，特別是 IoU@0.75。

背景問題：

`002_active_only_pca_autocorr_8class_5fold` 的 IoU@0.50 F1 已達 `0.7083`，但 IoU@0.75 F1 只有 `0.3308`。這表示 rep count 和大致位置已經可用，但 start/end 邊界仍不夠準。另有一個問題是 `active-phase-span` 仍會把同一 set 內部的 inactive gap 包進處理範圍，不夠符合「不要使用休息資料」的要求。

假設：

- 改用 active-contiguous block 可以排除 set 內部 inactive gaps；
- PCA/autocorr 先決定候選 rep 數與大致週期；
- 在候選 boundary 附近搜尋 motion-energy local minima，可以改善 start/end 對齊；
- 調參應先用 segmentation-only grid，不直接反覆跑完整分類。

預計改動：

- 新增 `--block-source active-phase-contiguous`；
- 新增 `pca-autocorr-refined` segmentation method；
- 新增 boundary refinement 參數：
  - `--boundary-refine-search-fraction`
  - `--boundary-refine-energy-window`
- 正式輸出 `003_active_only_pca_autocorr_refined_8class_5fold`；
- 更新 `artifacts_rep_classification/RESULTS_INDEX.md`。

不做的事：

- 不做 active/rest detection；
- 不使用休息資料；
- 不加入 supervised boundary model；
- 不重跑低分 threshold 類方法；
- 不把新增的本機第 9 人資料混進和第 002 版的比較。

預期改善：

- rep IoU@0.75 F1 提升；
- phase split IoU@0.50 F1 小幅提升；
- predicted reps 數量仍接近 true reps。

評估指標：

- rep IoU@0.25 / 0.50 / 0.75 F1；
- phase IoU@0.25 / 0.50 / 0.75 F1；
- subject-wise 5-fold classification accuracy；
- per-exercise rep IoU heatmap；
- confusion matrix。

風險：

- `active-phase-contiguous` 仍依賴 phase 標註，屬於 active-only 診斷，不是完整即時 pipeline；
- motion-energy minima 對所有動作不一定一致；
- 分類 accuracy 不一定跟 boundary IoU 同步提升。

審核結果：

使用者要求「做 active-only 的 rep boundary refinement」，因此直接實作並跑正式結果。

實際結果：

先用 segmentation-only grid 比較後，選擇：

```text
block source = active-phase-contiguous
boundary refine search fraction = 0.25
boundary refine energy window = 51
```

正式結果 `003_active_only_pca_autocorr_refined_8class_5fold`：

```text
true reps = 2424
predicted reps = 2374
classified reps = 2327
rep IoU@0.25 F1 = 0.9287
rep IoU@0.50 F1 = 0.7182
rep IoU@0.75 F1 = 0.3622
exercise classification accuracy = 0.8414
exercise macro F1 = 0.8382
phase IoU@0.25 F1 = 0.6865
phase IoU@0.50 F1 = 0.4383
phase IoU@0.75 F1 = 0.1730
```

與第 002 版比較：

```text
rep IoU@0.50 F1: 0.7083 -> 0.7182
rep IoU@0.75 F1: 0.3308 -> 0.3622
phase IoU@0.50 F1: 0.4063 -> 0.4383
classification accuracy: 0.8459 -> 0.8414
```

結論：

這次 refinement 對 boundary 有小幅但實質改善，尤其 IoU@0.75。phase split 也因 rep boundary 稍微變好而提升。但分類 accuracy 略降，代表目前 refinement 主要改善邊界精度，不一定改善分類特徵。距離 90% rep IoU@0.50 仍有差距，下一步應轉向更強的 boundary refinement，例如 per-exercise duration prior、DTW template refinement 或 supervised boundary model。

輸出結果：

- `artifacts_rep_classification/003_active_only_pca_autocorr_refined_8class_5fold/`
- `artifacts_rep_classification/RESULTS_INDEX.md`

下一步：

建議下一個正式實驗是 `004_active_only_dtw_template_refinement/`：

- 仍不使用休息資料；
- 用第 003 版 boundary 作候選；
- 每個動作從 training subjects 建立 rep template；
- validation subject 只用 template score 微調 boundary；
- 主指標看 IoU@0.75 和 phase IoU@0.50 是否繼續提升。

## 2026-05-16：每位受試者 Rep Segmentation 準度輸出

日期：2026-05-16

狀態：implemented

目的：

讓第 003 版 active-only boundary refinement 可以直接回答「每個人 rep 切割準度是多少」，而不是只有 overall 或 per-exercise 結果。

改動：

- 在 `tools/evaluate_rep_segmentation_classification.py` 新增 subject-wise rep segmentation metric 輸出；
- 每個 subject 分別計算 IoU@0.25 / 0.50 / 0.75 的 precision、recall、F1、matched reps、FP、FN、mean matched IoU；
- 新增 subject-wise heatmap：`rep_segmentation_iou_f1_by_subject.png`；
- 重跑 `003_active_only_pca_autocorr_refined_8class_5fold`，維持和第 002 版相同的 8 位 subject，不混入新增本機資料。

主要觀察：

```text
overall rep IoU@0.50 F1 = 0.7182
overall rep IoU@0.75 F1 = 0.3622
best IoU@0.50 subjects = thomas0506, ziho0512workout, hsianshun0514workout
weak IoU@0.50 subjects = yoru0511workout, kevin0509workout, yanz0510workout
```

結論：

目前每個人的 rep count 大致可用，但嚴格 boundary 精度仍不足。IoU@0.50 已能看出人與人之間差異，IoU@0.75 則顯示 start/end 邊界還需要更強的 per-exercise 或 subject-adaptive refinement。

輸出結果：

- `artifacts_rep_classification/003_active_only_pca_autocorr_refined_8class_5fold/rep_segmentation_metrics_by_subject.csv`
- `artifacts_rep_classification/003_active_only_pca_autocorr_refined_8class_5fold/rep_segmentation_iou_f1_by_subject.png`

## 2026-05-16：Waveform Rep 切割準確率圖

日期：2026-05-16

狀態：implemented

目的：

補上使用者要求的「波形 rep 切割準確率圖」，讓每一組 set 都能同時看到 waveform、ground truth boundary、predicted boundary，以及該組 IoU@0.50 的 rep segmentation F1。

呈現調整：

- 不使用底色 shading；
- 上排與下排都呈現同一段 sample waveform；
- 上排只畫 ground truth boundary；
- 下排只畫 prediction boundary；
- ground truth 用藍色；
- prediction 用紅色；
- start 用實線，end 用虛線；
- 每張圖標題顯示 F1、precision、recall、TP/FP/FN、mean matched IoU；
- 另外輸出 subject、exercise、subject × exercise 的準確率圖。

實際結果：

```text
output = artifacts_rep_classification/004_waveform_rep_accuracy_003_active_only/
set plots = 210
true reps = 2424
predicted reps = 2374
matched reps at IoU@0.50 = 1711
set-assigned F1 = 0.7132
```

說明：

第 004 版是 visualization / per-set diagnostic，不是新的模型。正式 overall rep segmentation 數值仍以第 003 版 `summary.json` 為主；第 004 版的 F1 是把 prediction 指派回每組 set 後，方便對照 waveform 圖的診斷數值。

## 2026-05-16：Boundary Feature 診斷與 Exercise-aware Refinement

日期：2026-05-16

狀態：implemented

目的：

回應「目前還是不準，是否要回去看其他特徵」的問題。先不盲目調參，而是量化不同特徵和 ground truth rep boundary 的對齊誤差，再用診斷結果做一版 exercise-aware boundary refinement。

第 005 版診斷：

- 新增 `tools/analyze_rep_boundary_features.py`；
- 針對 2214 個 internal rep boundary 分析 feature local min / max；
- 輸出特徵對齊誤差、within 50 samples 比例、每個動作推薦特徵、特徵波形範例圖。

主要診斷結果：

```text
overall best feature = gyro_magnitude_min
gyro_magnitude_min median error = 36.5 samples
gyro_magnitude_min within 50 samples = 0.5930
db_shoulder_press best feature = transition_energy_max
db_bench_press best feature = pca_extreme_max
```

第 006 版實作：

- 新增 segment method：`pca-autocorr-feature-refined`；
- rep count 與大致位置仍用 PCA/autocorr；
- boundary refinement 改成 exercise-aware feature score：
  - `db_bench_press`：PCA extreme max；
  - `db_shoulder_press`：transition energy max；
  - `db_rdl`：PCA velocity min；
  - 其他動作：gyro magnitude min。

實際結果：

```text
003 rep IoU@0.50 F1 = 0.7182
006 rep IoU@0.50 F1 = 0.7353
003 rep IoU@0.75 F1 = 0.3622
006 rep IoU@0.75 F1 = 0.3968
003 phase IoU@0.50 F1 = 0.4383
006 phase IoU@0.50 F1 = 0.4654
003 classification accuracy = 0.8414
006 classification accuracy = 0.8456
```

動作別觀察：

```text
db_shoulder_press IoU@0.50 F1: 0.4972 -> 0.7081
db_bench_press IoU@0.50 F1: 0.5621 -> 0.5820
one_arm_db_row IoU@0.50 F1: 0.8525 -> 0.8164
db_biceps_curl IoU@0.50 F1: 0.6853 -> 0.6608
```

結論：

換特徵方向是對的，但固定規則還不夠。第 006 版證明 `shoulder_press` 這類弱項可以靠不同 feature 改善，但也讓部分原本穩定的動作退步。下一步要避免手寫固定 feature map，改成在 training subjects 上做 per-exercise feature selection，或訓練輕量 boundary probability model，再用 duration prior / dynamic programming 產生最終切點。

## 2026-05-16：第 007 版 Rep 內九軸特徵關聯度分析

日期：2026-05-16

狀態：implemented

目的：

回應「先找出每種動作在一個 rep 裡 IMU 會有的特徵，分析九軸排列組合特徵或 waveform 特徵與每個動作的關聯度，找出適用於所有人的特徵」。

改動：

- 新增 `tools/analyze_rep_feature_relevance.py`；
- 使用第 003 版的 `rep_segmentation_truth_matches.csv` 作為 ground-truth rep 來源；
- 每個 rep 從原始 whole-session CSV 取出 `ax`、`ay`、`az`、`gx`、`gy`、`gz`、`mx`、`my`、`mz`；
- 抽取 378 個 rep-level 特徵，包含 time-domain、frequency、Haar wavelet、norm、PCA variance ratio、axis correlation、rep duration；
- 用 ANOVA F、mutual information、Random Forest feature importance 和 subject-wise fold top-20 stability 做 composite ranking；
- 用 GroupKFold 做 sensor / feature group ablation，確保 validation subject 不會出現在 training；
- 輸出 paper-style 圖表：overall feature ranking、subject-wise feature stability、feature family importance、sensor group ablation、動作別 feature heatmap、dominant axis distribution、top-feature embedding、top-feature confusion matrix。

正式結果：

```text
output = artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/
input run = artifacts_rep_classification/003_active_only_pca_autocorr_refined_8class_5fold/
ground-truth reps = 2424
subjects = 8
exercises = 8
features = 378
best feature set = acc_gyro
best subject-wise accuracy = 0.8499
```

Sensor / feature group ablation：

```text
acc_gyro             0.8499 ± 0.0749
top20_stable         0.7943 ± 0.0912
acc_only             0.7837 ± 0.0226
all_9_axis_features  0.7824 ± 0.0455
top40_stable         0.7806 ± 0.0762
correlations_only    0.7078 ± 0.0976
wavelet_only         0.6711 ± 0.1151
mag_only             0.6112 ± 0.2111
magnitudes_only      0.6006 ± 0.0705
gyro_only            0.5899 ± 0.0575
pca_only             0.3771 ± 0.0532
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

每個動作的穩定特徵方向：

```text
db_bench_press: axis_ax__min / axis_ax__mean / axis_ax__median
db_biceps_curl: axis_ax__std / axis_ax__range / axis_ax__iqr / gyro_norm
db_rdl: axis_ax__max / axis_ax__mean / acc_norm
db_shoulder_press: acc_norm / axis_ax
db_squat: axis_az / corr__ax__gz / gyro spectral entropy
db_triceps_curl: axis_ay / corr__ax__ay
db_weighted_crunch: corr__ax__az / acc_norm spectral entropy
one_arm_db_row: axis_ax / acc_norm
```

結論：

目前資料上，動作分類最穩定的不是「九軸全部使用」，而是 accelerometer + gyroscope。magnetometer 在未做完整校正與配戴位置 normalization 前會拉低跨人泛化；PCA-only 也不足以代表動作類別，較適合用於週期估計、降噪或可視化。下一步如果要改善模型，應把第 007 版的結果用在 train-fold 內 feature selection：分類器使用 `acc_gyro` 主特徵，boundary refinement 使用第 005/006 版已證明有效的 gyro magnitude / transition energy / PCA extrema，再用 per-exercise duration prior 或 supervised boundary score 改善 rep IoU。

輸出結果：

- `artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/summary.json`
- `artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/rep_feature_relevance_scores.csv`
- `artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/top_features_by_exercise.csv`
- `artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/sensor_group_ablation_accuracy.png`
- `artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/top_rep_features_overall.png`
- `artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/feature_importance_by_exercise.png`
- `artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/top20_feature_confusion_matrix.png`

## 2026-05-16：第 008 版 Feature-pair Scatter 可分性診斷

日期：2026-05-16

狀態：implemented

目的：

回應「類似第 007 版，但 x/y 軸是兩種特徵」的想法。這一版不是新增最終分類模型，而是把每個 rep 投到兩個可解釋特徵形成的二維空間，檢查 8 個動作是否有自然分群，並量化每組 feature pair 的 subject-wise 分類能力。

改動：

- 新增 `tools/analyze_feature_pair_scatter.py`；
- 讀取第 007 版的 `rep_level_feature_table.csv`、`rep_level_feature_metadata.csv`、`rep_feature_relevance_scores.csv`；
- 選出 34 組 feature pair，包含：
  - 各 feature method group 的 top-2 pair；
  - top relevance 且低冗餘的 ranked pairs；
  - 每個動作 one-vs-rest stable-effect top pair；
  - acc + gyro / spectral / wavelet / correlation / PCA 的 mixed pairs；
- 每一組 pair 單獨做 8 動作分類，使用 subject-wise 5-fold；
- 每一組 pair 輸出 scatter plot 與 normalized confusion matrix；
- 額外輸出 overall score bar plot、per-exercise F1 dotplot、top pair scatter grid。

方法定位：

這種圖的合理性來自 HAR / wearable sensor 文獻常見的低維 feature-space visualization，例如 PCA scatter 或 t-SNE scatter。這裡的差異是我們不使用不可解釋的 embedding 軸，而是直接用兩個可解釋的 IMU-derived features 作為 x/y 軸。圖只作為 feature separability diagnostic；正式判斷仍看 subject-wise accuracy、macro-F1、per-exercise F1 與 confusion matrix。

正式結果：

```text
output = artifacts_rep_classification/008_feature_pair_scatter_8class/
input run = artifacts_rep_classification/007_rep_feature_relevance_9axis_8class_5fold/
ground-truth reps = 2424
subjects = 8
exercises = 8
feature pairs = 34
best pair = best_acc_vs_best_spectral
best feature x = axis_ax__mean
best feature y = axis_gz__spectral_entropy
best accuracy = 0.7116
best macro-F1 = 0.7122
```

Top feature pairs：

```text
best_acc_vs_best_spectral  accuracy 0.7116  macro-F1 0.7122
best_acc_vs_best_gyro      accuracy 0.6518  macro-F1 0.6517
best_acc_vs_best_wavelet   accuracy 0.6493  macro-F1 0.6451
ranked_pair_11             accuracy 0.6498  macro-F1 0.6426
best_acc_vs_best_corr      accuracy 0.6423  macro-F1 0.6351
acc_axis_time_top2         accuracy 0.6370  macro-F1 0.6321
```

結論：

單純用兩個特徵，最佳結果約 `0.7116` accuracy / `0.7122` macro-F1，明顯低於第 007 版 `acc_gyro` 多特徵組合的 `0.8499`。這代表 feature-pair scatter 很適合做論文中的可視化與診斷：它能指出哪些動作在少數特徵上已經分開、哪些動作重疊；但若目標是 90% 以上跨人泛化，仍需要多特徵模型、train-fold feature selection，以及更強的 rep boundary refinement。

輸出結果：

- `artifacts_rep_classification/008_feature_pair_scatter_8class/summary.json`
- `artifacts_rep_classification/008_feature_pair_scatter_8class/selected_feature_pairs.csv`
- `artifacts_rep_classification/008_feature_pair_scatter_8class/feature_pair_metrics.csv`
- `artifacts_rep_classification/008_feature_pair_scatter_8class/feature_pair_per_exercise_metrics.csv`
- `artifacts_rep_classification/008_feature_pair_scatter_8class/feature_pair_overall_scores.png`
- `artifacts_rep_classification/008_feature_pair_scatter_8class/feature_pair_per_exercise_f1_dotplot.png`
- `artifacts_rep_classification/008_feature_pair_scatter_8class/top_feature_pair_scatter_grid.png`
- `artifacts_rep_classification/008_feature_pair_scatter_8class/scatter_pairs/`
- `artifacts_rep_classification/008_feature_pair_scatter_8class/confusion_matrices/`

## 2026-05-16：第 009 版 Universal Rep Boundary 訊號分析

日期：2026-05-16

狀態：implemented

目的：

回應「目前是否知道要如何切割，如果還不知道，先跑波形分析找要用什麼特徵切 rep」。這一版專門分析未知動作時可用的泛化切割訊號，不使用動作分類特徵當第一刀。

改動：

- 新增 `tools/analyze_universal_rep_boundary_signals.py`；
- 讀取第 003 版 ground-truth rep boundary；
- 針對 2214 個 internal rep boundary，比較多種 waveform 特徵的 local min / max 是否貼近真實切點；
- 比較 smooth window `9 / 21 / 51` 與 energy window `21 / 51 / 81`；
- 額外分析每個 active set 的週期估計：用 autocorrelation / FFT 比較 `pca_motion`、`acc_magnitude`、`gyro_magnitude`、`motion_energy` 等訊號和真實 rep period 的誤差；
- 輸出 universal boundary ranking、跨動作 heatmap、period estimation summary、waveform example plots。

正式結果：

```text
output = artifacts_rep_classification/009_universal_rep_boundary_signal_analysis/
input run = artifacts_rep_classification/003_active_only_pca_autocorr_refined_8class_5fold/
internal GT boundaries = 2214
boundary feature rows = 119556
period rows = 2898
best universal boundary feature = gyro_magnitude_min_s9
boundary median abs error = 36.5 samples
boundary within 50 samples = 0.5930
boundary within 100 samples = 0.8921
best period signal = pca_motion
best period method = autocorr
period median abs error = 7.0 samples
period median relative abs error = 0.0217
period within 10% = 0.8696
```

Top universal boundary features：

```text
gyro_magnitude_min_s9        median 36.5  within50 0.5930
gyro_magnitude_min_s21       median 39.0  within50 0.5786
gyro_magnitude_min_s51       median 40.0  within50 0.5682
pca_velocity_min_s21_e21     median 50.0  within50 0.5045
pca_velocity_min_s9_e21      median 48.0  within50 0.5158
gyro_jerk_max_s51_e21        median 53.0  within50 0.4846
```

每個動作在最佳 universal boundary feature 的 within-50 表現：

```text
db_biceps_curl      0.7824
db_squat            0.6597
db_triceps_curl     0.6452
one_arm_db_row      0.6444
db_weighted_crunch  0.5788
db_bench_press      0.5243
db_rdl              0.4888
db_shoulder_press   0.4143
```

結論：

目前已知道比較合理的切割方向：

```text
1. 用 pca_motion + autocorrelation 估週期 / rep count；
2. 用 gyro_magnitude_min_s9 找週期候選附近的 boundary valley；
3. 用 duration prior / monotonic constraint / dynamic programming 避免切太細或邊界跳動；
4. 初切後再抽第 007 版分類特徵辨識動作；
5. 知道動作後再做 second-pass exercise-aware boundary refinement 和 phase split。
```

但第 009 版也證明「只靠一個 universal 特徵」還不夠。最佳 boundary 特徵 within-50 只有 `0.5930`，因此下一步不應再單純找單一特徵，而應把 `pca_motion` 週期 prior 和 `gyro_magnitude` valley 整合成一個 sequence-level segmenter。

輸出結果：

- `artifacts_rep_classification/009_universal_rep_boundary_signal_analysis/summary.json`
- `artifacts_rep_classification/009_universal_rep_boundary_signal_analysis/universal_boundary_feature_ranking.csv`
- `artifacts_rep_classification/009_universal_rep_boundary_signal_analysis/universal_boundary_feature_ranking.png`
- `artifacts_rep_classification/009_universal_rep_boundary_signal_analysis/universal_boundary_within_50_by_exercise.png`
- `artifacts_rep_classification/009_universal_rep_boundary_signal_analysis/period_estimation_summary.csv`
- `artifacts_rep_classification/009_universal_rep_boundary_signal_analysis/period_estimation_error_by_signal.png`
- `artifacts_rep_classification/009_universal_rep_boundary_signal_analysis/universal_feature_waveform_examples/`

## 2026-05-16：第 010 版 Universal Periodic Gyro-Valley Segmenter

日期：2026-05-16

狀態：implemented

目的：

回應「根據第 009 版方法試切 rep，輸出切割波形圖與切割正確率，並將每個動作的正確率做成一張表」。

改動：

- 在 `tools/evaluate_rep_segmentation_classification.py` 新增 `pca-autocorr-gyro-valley` segment method；
- 用 `pca_motion + autocorrelation` 估 active set 的 dominant period 和 expected rep count；
- 在 expected boundary 附近搜尋 `gyro_magnitude` valley；
- 加入 duration prior、rep-count prior、max reps prior，避免錯誤週期造成 over-segmentation；
- 新增 `--skip-classification`，可先只跑 rep segmentation / phase split；
- 新增 `rep_segmentation_accuracy_by_exercise_table.csv` 與 `.png`，把每個動作的 IoU 正確率整理成表格；
- 加速讀檔與標籤擷取：只讀必要欄位，並用 vectorized / numpy indexing 取代逐列 pandas `.iloc`；
- 更新 `tools/plot_waveform_rep_accuracy.py`，產生第 010 版每組上下排波形切割圖。

正式結果：

```text
output = artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold/
waveform output = artifacts_rep_classification/010_waveform_rep_accuracy_universal_periodic_gyro_valley/

truth reps = 2720
predicted reps = 2740

rep IoU@0.25 F1 = 0.9092
rep IoU@0.50 F1 = 0.7278
rep IoU@0.75 F1 = 0.3949
rep IoU@0.85 F1 = 0.2564
rep IoU@0.90 F1 = 0.1626
rep IoU@0.95 F1 = 0.0670

phase IoU@0.50 F1 = 0.4552
phase IoU@0.90 F1 = 0.0432
waveform set-level IoU@0.50 F1 = 0.7234
waveform set plots = 236
```

每個動作 IoU@0.50 F1：

```text
db_squat            0.8413
one_arm_db_row      0.8407
db_weighted_crunch  0.7964
db_rdl              0.7778
db_triceps_curl     0.7720
db_shoulder_press   0.6589
db_biceps_curl      0.6133
db_bench_press      0.5499
```

結論：

第 010 版證明「PCA 估週期 + gyro valley 精修」可以形成可用的 universal active-only rep segmenter，IoU@0.50 F1 為 `0.7278`。這和第 006 版 exercise-aware refinement 的 `0.7353` 接近，但第 010 版第一刀不依賴動作類別，因此更適合未知 waveform 流程。

目前仍未達 90% F1，也遠未達 IoU@0.90 的高精度切割目標。弱項集中在 `db_bench_press`、`db_biceps_curl`、`db_shoulder_press`，下一步應在初切後先做動作分類，再根據分類結果做 second-pass exercise-aware boundary refinement。

輸出結果：

- `artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold/summary.json`
- `artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold/rep_segmentation_accuracy_by_exercise_table.png`
- `artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold/rep_segmentation_iou_metrics.png`
- `artifacts_rep_classification/010_universal_periodic_gyro_valley_8class_5fold/rep_segmentation_iou_f1_by_exercise.png`
- `artifacts_rep_classification/010_waveform_rep_accuracy_universal_periodic_gyro_valley/summary.json`
- `artifacts_rep_classification/010_waveform_rep_accuracy_universal_periodic_gyro_valley/waveform_rep_accuracy_by_subject.png`
- `artifacts_rep_classification/010_waveform_rep_accuracy_universal_periodic_gyro_valley/waveform_rep_accuracy_by_exercise.png`
- `artifacts_rep_classification/010_waveform_rep_accuracy_universal_periodic_gyro_valley/sets_all/`

## 2026-05-16：第 011 版 Multi-feature Boundary Scoring

日期：2026-05-16

狀態：implemented

目的：

回應「前面 rep IoU@0.90 必須很高，切 rep 時需要更多特徵進來」。這一版嘗試把第 010 版的單一 `gyro_magnitude` valley 規則改成 supervised boundary candidate scorer，目標是提高高 IoU 門檻下的 rep boundary 準確率。

改動：

- 新增 `tools/evaluate_multifeature_boundary_score.py`；
- 使用 `active-phase-contiguous`，只處理已在運動中的區段，不納入 rest；
- 先用 `pca_motion + autocorrelation` 估 set-level period / expected rep count；
- 在每個預期切點附近收集多種候選 boundary：
  - PCA min / max；
  - abs PCA max；
  - PCA velocity / acceleration；
  - accelerometer magnitude min / max；
  - gyroscope magnitude min / max；
  - acc / gyro jerk；
  - transition energy；
  - motion energy；
  - dominant-axis min / max；
- 對候選點抽局部 value、mean、std、left/right mean、left-right difference 等特徵；
- 用 subject-wise 5-fold 訓練 boundary classifier，validation subject 不出現在 training；
- 以 duration prior、rep-count prior、prior-distance penalty 和 monotonic sequence constraint 選出整組 boundaries；
- 輸出 IoU@0.50 / 0.75 / 0.85 / 0.90 / 0.95、每個動作的 IoU@0.90 表格、internal boundary error，以及上下排波形切割圖。

正式結果：

```text
output = artifacts_rep_classification/011_multifeature_boundary_score_high_iou/
waveform output = artifacts_rep_classification/011_waveform_rep_accuracy_multifeature_boundary_score/

truth reps = 2720
predicted reps = 2658

rep IoU@0.50 F1 = 0.7382
rep IoU@0.75 F1 = 0.4106
rep IoU@0.85 F1 = 0.2510
rep IoU@0.90 F1 = 0.1621
rep IoU@0.95 F1 = 0.0621

median internal boundary error = 60.0 samples = 600.0 ms
internal boundary within 10 samples = 0.1188
internal boundary within 20 samples = 0.2246

phase IoU@0.50 F1 = 0.4736
phase IoU@0.90 F1 = 0.0356
waveform set plots = 236
```

每個動作 IoU@0.90 F1：

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

結論：

第 011 版沒有達到 90% 高 IoU 目標，也不應取代第 010 版作為目前主方法。它的 IoU@0.50 F1 從第 010 版的 `0.7278` 小幅到 `0.7382`，但 IoU@0.90 F1 只有 `0.1621`，代表「大致分到同一個 rep」有改善，但「boundary 精準貼到 GT」沒有改善。

主要問題推論：

- 候選點雖多，但真實 boundary 附近 5-10 samples 的候選 recall 可能不足；
- positive radius 10 samples 對 classifier 來說仍太寬，無法直接優化 IoU@0.90；
- logistic scorer 只能做局部候選分類，沒有真正學到整段 rep 相位對齊；
- 不同動作的 boundary cue 差異很大，單一跨動作 boundary scorer 會平均掉特徵；
- 沒有使用新人少量標註做 subject-specific calibration。

下一步：

下一版建議不要再單純加特徵，而是做 `012_candidate_recall_and_template_alignment`：

1. 先量化每個方法在 GT boundary ±5 / ±10 / ±20 samples 內是否有候選點，確定上限；
2. 用第 010 版初切後的 reps 建立 per-exercise normalized template；
3. 對每個 set 做 DTW / phase-normalized template alignment，修正 internal boundary；
4. 加入 subject adaptation：新人少量標註後估計個人 duration scale、axis weight、boundary offset；
5. 再用 IoU@0.90、boundary median error、within-10-sample rate 重新比較第 010 / 第 011 / 第 012。

輸出結果：

- `artifacts_rep_classification/011_multifeature_boundary_score_high_iou/summary.json`
- `artifacts_rep_classification/011_multifeature_boundary_score_high_iou/rep_segmentation_iou_0.90_by_exercise_table.png`
- `artifacts_rep_classification/011_multifeature_boundary_score_high_iou/boundary_error_by_exercise.png`
- `artifacts_rep_classification/011_method_comparison_high_iou/rep_segmentation_methods_iou_0.90.png`
- `artifacts_rep_classification/011_method_comparison_high_iou/rep_segmentation_exercise_delta_iou_0.90.png`
- `artifacts_rep_classification/011_waveform_rep_accuracy_multifeature_boundary_score/waveform_rep_accuracy_by_subject.png`
- `artifacts_rep_classification/011_waveform_rep_accuracy_multifeature_boundary_score/waveform_rep_accuracy_by_exercise.png`
- `artifacts_rep_classification/011_waveform_rep_accuracy_multifeature_boundary_score/sets_all/`
