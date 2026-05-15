# Rep 切割、動作辨識與 TUT 分析實驗規劃

## 目的

本文件用來固定後續實作方向。每次實作前，先在此文件對齊：

1. 要比較哪些方法；
2. 為什麼選這些方法；
3. 預期改善目前哪個問題；
4. 如何用數值與圖表證明改善；
5. 是否適合部署到 Luckfox Pico Zero。

目前不直接追求把所有方法一次做完，而是先建立一組可解釋、可比較、可部署的 baseline，再逐步加入創新方法。

## 目前問題

目前已經有：

- `dominant-axis`
- `short-time-energy`
- `pca-extrema`
- `pca-autocorr`
- `pca-extrema-fft`
- oracle `labels`

主要問題是：

- peak 類方法容易 over-segmentation；
- 單純 FFT 只能估計整段週期或總次數，不容易給出每一下 rep 的精準 start/end；
- 現有方法還沒有利用「已知動作種類」來調整切割參數；
- 目前 phase split 還沒有正式獨立評估，因此 TUT、向心/離心比例還不能當穩定訓練建議；
- 現有結果尚不能宣稱超越高品質文獻，尤其 DS-MS-TCN 類 sequence segmentation 模型的 IoU F1 明顯更高。

## 任務優先度

### P0. 大動作 / set detection

優先原因：

- 如果運動區間沒有先切準，rep segmentation 會把 rest、移動手腕、拿放啞鈴切成假 rep；
- LEAN 也強調 repetition-counting algorithm 應該只在使用者確實正在運動時啟動，這能降低 false positive。

預計方法：

- sliding window activity detector；
- energy / variance gate；
- exercise classifier confidence gate；
- minimum active duration + rest hysteresis。

目前驗證結果：

```text
target=action, method=oracle-action:
sample F1 = 1.0000
segment IoU@0.50 F1 = 1.0000
true segments = 68
predicted segments = 68

target=action, method=imu-energy:
sample F1 = 0.4301
segment IoU@0.50 F1 = 0.0000
true segments = 68
predicted segments = 3956

target=action, method=imu-hysteresis:
sample F1 = 0.6357
segment IoU@0.50 F1 = 0.1429
true segments = 68
predicted segments = 114

target=action, method=window-rf:
sample F1 = 0.8654
segment IoU@0.50 F1 = 0.1978
true segments = 68
predicted segments = 23
```

解讀：

- `oracle-action` 確認資料內的 action 標註可以形成 set-level active ground truth；
- 單純 `imu-energy` / `imu-variance` 會嚴重 fragmentation，已不再作為預設實驗方法；
- `imu-hysteresis` 能降低 fragmentation，但 segment IoU 仍低；
- `window-rf` 的 sample F1 已到 `0.8654`，代表 active/rest window 特徵可學；但 segment F1 只有 `0.1978`，代表 set boundary post-processing 與 action label 定義仍是瓶頸；
- 因此目前 rep segmentation 要接近 90% IoU，不能只調 peak detection，必須先建立更穩定的 active/set detector 與明確的 set boundary ground truth。

### P1. 動作辨識

優先原因：

- 不同動作的週期、幅度、主要運動軸、合理 rep duration 不同；
- 先知道動作，後續才能用 per-exercise prior 做切割；
- 也能輸出論文常見的 confusion matrix。

預計方法：

- repetition-level RandomForest baseline；
- set-level majority vote；
- optional tiny TCN / 1D CNN 作為 supervised 上限模型。

### P2. Rep boundary segmentation

優先原因：

- rep boundary 是後續 TUT、ROM、速度、疲勞趨勢的基礎；
- 目前 IoU@0.50 F1 還偏低，最需要改善。

預計方法：

- PCA + peak detection；
- PCA + autocorrelation-constrained peak detection；
- PCA + wavelet denoising / scalogram energy；
- DTW template refinement；
- tiny TCN boundary detector。

### P3. 向心 / 離心 phase split

優先原因：

- TUT 和重訓建議需要 phase duration；
- phase split 應該在 rep boundary 穩定後再做，否則錯誤會被放大。

預計方法：

- state machine；
- midpoint / reversal point；
- velocity sign / derivative zero-crossing；
- HMM 或 tiny TCN 作為 supervised phase model。

### P4. 重訓建議

優先原因：

- 使用者最終需要的是訓練回饋，不只是模型分數；
- 但建議必須建立在穩定 segmentation 與 phase split 上。

預計輸出：

- 每組總 reps；
- 平均 rep duration；
- concentric / eccentric ratio；
- TUT per rep / per set；
- ROM proxy；
- rep tempo consistency；
- velocity drop / fatigue trend；
- 是否過快、過慢、節奏不穩、ROM 下降。

## 第一輪建議比較方法

第一輪不建議做 Transformer。原因是目前資料量有限，而且 Luckfox Pico Zero 雖有 NPU，但 CPU/RAM 資源仍有限；Transformer 對即時部署和除錯成本都太高。

| 類別 | 方法 | 優先度 | 角色 | 主要優點 | 主要風險 |
|---|---|---:|---|---|---|
| 傳統 baseline | FFT count baseline | 中 | 只比較 count-level | 可解釋、便宜 | 不適合精準 boundary |
| 傳統 baseline | Dominant-axis peak | 高 | 低成本 baseline | 即時、可部署 | 對配戴方向敏感 |
| 傳統 baseline | Short-time energy | 中 | 能量法 baseline | 對強度變化敏感 | 容易受雜訊與休息動作影響 |
| 核心 baseline | PCA + peak | 很高 | 目前主要 baseline | 不必人工選軸 | over-segmentation |
| 改良方法 | PCA + autocorrelation + peak | 很高 | 已實作改善方法 | 用週期限制 peak distance | 變速或不穩節奏會失準 |
| 改良方法 | PCA + wavelet + peak | 高 | 非穩態訊號改善 | 適合速度變化與疲勞 | 參數較多 |
| 個人化方法 | DTW template refinement | 高 | 少量標註後改善新人 | 適合形狀比對與 few-shot | 計算量較高 |
| Phase 方法 | State machine | 很高 | 向心/離心切割 | 可解釋、低成本 | 規則需依動作調整 |
| Sequence model | HMM | 中高 | phase sequence baseline | 能建模狀態轉移 | 需要穩定 observation features |
| Sequence model | Tiny TCN | 高 | supervised 上限模型 | 可學 boundary / phase | 需要更多標註與量化部署 |
| 大模型 | Transformer | 低 | 暫不做 | 複雜長序列能力強 | 資料量與硬體成本不划算 |

## 我們的創新方向

### 方向 A：Exercise-aware PCA Autocorrelation Peak

流程：

```text
set detection
-> exercise recognition
-> robust z-score
-> PCA principal motion signal
-> autocorrelation estimate dominant period
-> per-exercise duration prior
-> constrained peak/trough candidates
-> boundary midpoint
```

預期改善：

- 降低目前 PCA peak 的過切；
- 比純 FFT 更能保留 local boundary；
- 比 dominant-axis 更不依賴配戴方向。

主要評估：

- IoU@0.50 F1；
- predicted / true reps ratio；
- per-exercise matched rate；
- set-level count MAE；
- waveforms with boundary lines。

目前結果：

```text
method: pca-autocorr
true reps: 2424
predicted reps: 4846
IoU@0.50 precision: 0.2070
IoU@0.50 recall: 0.4138
IoU@0.50 F1: 0.2759
IoU@0.75 F1: 0.0930
classification accuracy: 0.8482
```

與 `pca-extrema-fft` 比較：

```text
predicted reps: 5942 -> 4846
IoU@0.50 precision: 0.1439 -> 0.2070
IoU@0.50 recall: 0.3527 -> 0.4138
IoU@0.50 F1: 0.2044 -> 0.2759
IoU@0.75 F1: 0.0304 -> 0.0930
```

初步結論：

自相關週期限制比目前 FFT 週期限制更適合這批資料，主要改善是降低 over-segmentation 並提升高 IoU 門檻下的匹配數。但 IoU@0.50 F1 仍只有 `0.2759`，下一步仍需要 active set detection、per-exercise prior 或 phase-aware refinement。

### 方向 B：Wavelet-Denoised PCA Boundary

流程：

```text
set detection
-> PCA principal motion signal
-> wavelet denoising or CWT energy ridge
-> local extrema candidates
-> autocorrelation period constraint
-> boundary selection
```

預期改善：

- 對非穩態重訓動作更穩，例如疲勞造成速度下降；
- 比 STFT 更適合 rep 速度變化。

主要風險：

- wavelet family、scale range、threshold 需要系統化調參；
- Luckfox 即時端可能不適合完整 CWT，因此可先放在 post-set 分析。

### 方向 C：Few-shot DTW Personalization

流程：

```text
new subject first 1-2 labeled sets
-> build per-exercise rep template
-> later sets generate candidate segments
-> DTW score refine candidate boundary
-> update subject-specific period / amplitude prior
```

預期改善：

- 新人資料少時仍能快速個人化；
- 對不同人節奏、手腕姿勢、動作幅度更有彈性。

主要風險：

- DTW 對所有位置暴力搜尋太貴；
- 需要先用 peak/autocorrelation 產生少量候選，再用 DTW 精修。

### 方向 D：State Machine Phase Split

流程：

```text
rep boundary
-> within-rep PCA signal / velocity proxy
-> reversal point
-> eccentric / concentric state assignment
-> reject impossible transitions
-> TUT and phase ratio
```

預期改善：

- 向心/離心切割可解釋；
- 能直接支援 TUT、tempo、ROM 趨勢分析；
- 適合即時端低成本部署。

主要風險：

- 不同動作向心/離心方向不同，需要 exercise-aware mapping；
- 若 rep boundary 錯，phase split 會跟著錯。

## 即時與做完一組的分工

Luckfox Pico Zero 主要規格：Cortex-A7 1.2GHz、256MB DDR3L、NPU 最高約 1 TOPS，支援 INT4/INT8/INT16。這代表它適合輕量 DSP、tree model、tiny CNN/TCN，不適合大型 Transformer。

### 即時端

目標：

- 即時偵測正在運動；
- 粗略辨識動作；
- 粗略 rep count；
- 粗略 tempo / TUT 提醒；
- 避免明顯錯誤，例如節奏過快、停頓太久。

建議方法：

- energy gate；
- exercise classifier；
- PCA + autocorrelation-constrained peak；
- state machine phase split；
- tiny quantized model only if needed。

### 做完一組後

目標：

- 精修 rep boundaries；
- 計算正式 TUT、phase ratio、ROM proxy；
- 產生圖表與報告；
- 給出比較完整的訓練建議。

建議方法：

- wavelet refinement；
- DTW template matching；
- HMM / tiny TCN offline refinement；
- set-level trend analysis。

## 評估指標

### Rep segmentation

- IoU@0.25 / 0.50 / 0.75 Precision、Recall、F1；
- mean matched IoU；
- boundary start error / end error，單位秒；
- predicted / true reps ratio；
- per-set count MAE；
- within +/-1 rep count accuracy。

### 動作辨識

- accuracy；
- macro F1；
- weighted F1；
- confusion matrix；
- per-exercise precision / recall / F1；
- leave-subject-out 或 subject-wise K-fold。

### 向心 / 離心

- phase boundary error，單位秒；
- phase IoU；
- concentric duration MAE；
- eccentric duration MAE；
- concentric/eccentric ratio error；
- TUT per rep / per set error。

### 部署

- latency per window；
- memory usage；
- CPU utilization；
- model size；
- battery / sustained runtime proxy；
- whether it can run without storing the full session。

## 結果呈現

### 表格

1. Overall method comparison：
   - rows: 方法；
   - cols: IoU@0.50 F1、count MAE、prediction ratio、latency。

2. Per-exercise segmentation table：
   - rows: 動作；
   - cols: 各方法 IoU@0.50 F1。

3. Phase/TUT table：
   - rows: 動作；
   - cols: TUT MAE、concentric MAE、eccentric MAE、ratio error。

### 圖

1. Confusion matrix：
   - 動作辨識結果；
   - row-normalized 版本一起輸出。

2. Method comparison bar chart：
   - IoU@0.50 F1；
   - count MAE；
   - over-segmentation ratio。

3. Per-exercise heatmap：
   - 每個動作對每個方法的切割正確率。

4. Waveform boundary comparison：
   - 同一組 set；
   - ground truth 綠線；
   - prediction 橘線；
   - 實線 start；
   - 虛線 end；
   - 不塗底色。

5. TUT / phase ratio plot：
   - 每一 rep 的 concentric/eccentric duration；
   - 每組平均與標準差；
   - 顯示是否節奏漂移或疲勞。

## 實作審核流程

每次實作前都要先列：

```text
目的：
假設：
要改的檔案：
方法：
預期改善：
評估指標：
可能風險：
是否影響現有 artifacts：
是否需要重跑完整 pipeline：
```

使用者審核後才實作。

## 文獻引用與用途

[1] M. Shang et al., "DS-MS-TCN: Otago Exercises Recognition With a Dual-Scale Multi-Stage Temporal Convolutional Network," IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 12, pp. 7138-7150, 2024, doi: 10.1109/JBHI.2024.3455426.

用途：近期 IEEE JBHI sequence segmentation 代表作。其 micro/macro label 與 IoU F1 呈現方式適合當我們的上限模型與結果圖參考。

[2] J. W. Dorschky et al., "LEAN: Real-Time Analysis of Resistance Training Using Wearable Computing," Sensors, vol. 23, no. 10, 4602, 2023, doi: 10.3390/s23104602.

用途：最接近本專案的即時 resistance training wearable 系統，包含 rep counting、form classification、向心/離心時間比例與低記憶體設計。

[3] T. T. de Beukelaar and D. Mantini, "Monitoring Resistance Training in Real Time with Wearable Technology: Current Applications and Future Directions," Bioengineering, vol. 10, no. 9, 1085, 2023, doi: 10.3390/bioengineering10091085.

用途：近期 review，說明 wearable 在 resistance training 即時監測中的應用與限制，可支撐研究動機。

[4] D. Morris, T. S. Saponas, A. Guillory, and I. Kelner, "RecoFit: Using a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises," in Proc. CHI, 2014, pp. 3225-3234, doi: 10.1145/2556288.2557116.

用途：經典 wearable strength-training pipeline，將任務拆成 exercise period segmentation、recognition、rep counting，並使用自相似/週期概念。

[5] S. Ishii, A. Yokokubo, M. Luimula, and G. Lopez, "ExerSense: Physical Exercise Recognition and Counting Algorithm from Wearables Robust to Positioning," Sensors, vol. 21, no. 1, 91, 2021, doi: 10.3390/s21010091.

用途：correlation / template-based 方法代表，適合支持 few-shot template 或 DTW personalization 的設計。

[6] X. Guo, J. Liu, and Y. Chen, "When Your Wearables Become Your Fitness Mate," Smart Health, vol. 16, 100114, 2020, doi: 10.1016/j.smhl.2020.100114.

用途：運動 review / recommendation 系統，支持把 motion strength、speed、exercise score 轉成訓練回饋。

[7] S. Zelman, M. Dow, T. Tabashum, T. Xiao, and M. V. Albert, "Accelerometer-Based Automated Counting of Ten Exercises without Exercise-Specific Training or Tuning," Journal of Healthcare Engineering, 2020, 8869134, doi: 10.1155/2020/8869134.

用途：FFT、threshold crossing、low-pass threshold 的傳統 count baseline，適合和我們的 PCA/autocorrelation 方法比較。

[8] B. J. Schoenfeld, D. I. Ogborn, and J. W. Krieger, "Effect of Repetition Duration During Resistance Training on Muscle Hypertrophy: A Systematic Review and Meta-Analysis," Sports Medicine, vol. 45, pp. 577-585, 2015, doi: 10.1007/s40279-015-0304-0.

用途：TUT / tempo 建議的生理依據。它提醒我們 TUT 可作為節奏與訓練品質指標，但不應過度宣稱單一 TUT 長度必然帶來更好肌肥大。
