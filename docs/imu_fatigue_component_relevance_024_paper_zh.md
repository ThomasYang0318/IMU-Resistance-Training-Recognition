# 024 IMU 疲勞相關成分結果圖：論文撰寫版

## 建議標題

以 IMU 與 VO2 特徵量化阻力訓練中疲勞相關動作變化之初步驗證

## 摘要

本研究的長期目標是建立一套可在阻力訓練中提供即時或半即時訓練建議的穿戴式感測系統。若要讓系統不只是計算 rep 次數，而能進一步估計訓練疲勞狀態，必須先驗證 IMU 波形特徵是否和主觀疲勞程度具有可量化關聯。因此，本節使用已人工標註的 rep 與 concentric/eccentric phase 作為 ground truth，排除自動切割誤差的干擾，分析 9 軸 IMU 特徵、time under tension (TUT)、CE phase 動作型態變化與 Borg/RPE 之間的關聯。此外，本研究也納入 VO2 延遲反應，檢查呼吸代謝訊號是否能作為輔助生理負荷指標。結果顯示，累積 TUT、CE phase waveform range、phase similarity drift、concentric gyroscope variation、phase movement rate 與延遲 VO2 皆與 Borg/RPE 呈現不同程度關聯。其中，累積 TUT 與 Borg/RPE 的 raw Spearman correlation 為 0.4594，且在同人同動作中心化後提升至 0.6010；CE phase range、phase similarity drift 與 phase movement rate 也呈現中低度但可解釋的相關。這些結果支持 IMU 並非直接量測肌肉疲勞本身，而是可量化和疲勞相關的動作學變化，並且後續模型應採用 exercise-aware 與 phase-aware 的設計。

## 1. 研究背景與動機

阻力訓練中的疲勞監測對於訓練品質、傷害風險控制與個人化訓練建議具有重要意義。傳統上，肌肉疲勞可透過 surface electromyography (sEMG)、血乳酸、力量下降或速度下降等方式評估，但這些方法在一般健身或可攜式系統中存在設備成本、量測便利性或情境限制。近年穿戴式感測器逐漸被用於阻力訓練監測，IMU 可提供加速度、角速度與姿態相關訊號，適合用於 rep segmentation、動作辨識、速度變化與動作品質分析 [1]。

然而，本研究要回答的問題不是「IMU 是否可以直接量到肌肉疲勞」。肌肉疲勞本身是神經肌肉與代謝系統共同作用的生理現象，IMU 量到的是外部動作學訊號。因此，較合理的研究定位應該是：

```text
IMU can quantify fatigue-related movement changes.
```

也就是說，IMU 可量化疲勞發生時可能伴隨出現的動作變化，例如 rep 時間拉長、phase waveform range 改變、角速度變化上升、動作相似度下降或 concentric/eccentric 節奏改變。這個定位與 Chan 等人使用 IMU 建立 subject-specific movement composite index 來偵測 fatigue-related kinematic changes 的想法一致 [5]。

另一方面，本研究使用 Borg/RPE 作為疲勞感標籤。Borg/RPE 是主觀感受量表，並非直接生理量測，但文獻顯示 RPE 在阻力訓練中可反映一定程度的肌肉疲勞與生理反應。Zhao 等人指出，在單關節阻力訓練中，RPE 與 sEMG-based spectral fatigue index 具有關聯，並可用於預測疲勞變化 [2]；在 back squat 研究中，RPE 與 sEMG 疲勞指標也呈現顯著關聯，但 velocity loss 在非爆發式訓練情境下不一定能正確反映疲勞 [3]。因此，本研究選擇 Borg/RPE 作為初步 fatigue target 是合理的，但論文中應清楚說明其主觀性與限制。

## 2. 為什麼要跑第 024 張結果圖

第 024 張結果圖的目的不是展示最終模型準確率，而是作為「建模前的特徵合理性驗證」。在進一步建立 RPE 預測模型或訓練建議系統之前，需要先回答三個問題。

第一，IMU 波形特徵是否真的和 Borg/RPE 有關。若使用 ground-truth rep 與 CE phase 標記後，IMU 特徵仍完全無法和 RPE 產生關聯，代表後續即使改善自動切割，也不一定能幫助疲勞估計。因此，本圖先在理想切割條件下檢查可學訊號上限。

第二，疲勞相關訊號應該看整段 rep，還是要拆成 concentric/eccentric phase。阻力訓練的 rep 不是單一均質動作，向心與離心階段的肌肉收縮型態、速度控制與感測器波形皆可能不同。若只看整段 rep 的平均特徵，可能會混掉 phase-specific fatigue patterns。因此，本圖加入 CE phase range、phase similarity、concentric gyro 與 CE ratio drift，檢查 phase-aware feature 是否比單純時間或整段波形更有研究價值。

第三，不同動作是否能共用同一個疲勞規則。若所有動作都呈現相同特徵變化，例如「向心時間變長就代表疲勞」，模型可以做成單一規則；但若不同動作的特徵相關性不同，就需要 exercise-aware 設計。第 024 圖的 heatmap 正是用來檢查這件事。

因此，第 024 圖在論文中的角色是：

```text
先證明 IMU / VO2 特徵和 Borg/RPE 之間存在可解釋關聯，再合理化後續 exercise-aware phase fatigue model 的設計。
```

## 3. 方法

### 3.1 資料與標註

本分析使用阻力訓練資料集中已標註的 workout sessions。為了避免自動 rep segmentation 誤差影響疲勞特徵分析，本階段使用 ground-truth rep boundary 與 ground-truth concentric/eccentric phase 標記。Borg/RPE target 來自每位受試者對應的 workbook，其中 `X` 表示該 rep 未完成，不納入 RPE target 訓練；空白欄位表示沿用前一個 REP/RPE 值。

本分析分成兩個資料層級：

```text
rep-level IMU/RPE analysis:
n = 1677 reps
subjects = haoyu, hsianshun, tsenyu, yanz, yoru, yushuan

set-level CE phase/RPE analysis:
n = 143 sets
subjects = haoyu, hsianshun, tsenyu, yanz, yoru, yushuan

VO2 lag analysis:
n = 96 sets
subjects = haoyu, yanz, yoru, yushuan
```

VO2 只在同時具備 RPE workbook 與 VO2 time alignment 的受試者中分析。由於 VO2 是呼吸代謝訊號，反應時間會落後於單一下 rep，因此本研究不將 VO2 視為即時 rep label，而是以 set 後 lag window 的方式評估其與 RPE 的關係。

### 3.2 IMU 特徵設計

IMU 輸入包含 9 軸訊號：

```text
ax, ay, az, gx, gy, gz, mx, my, mz
```

特徵設計分為四類。

第一類是累積負荷與時間特徵，例如 cumulative active time、cumulative TUT、concentric duration、eccentric duration 與 CE ratio。TUT 被納入是因為阻力訓練中肌肉承受張力的時間會影響訓練刺激與主觀疲勞感；Burd 等人亦指出 time under tension 會影響 resistance exercise 後的肌肉蛋白合成反應 [8]。

第二類是 phase waveform range。每一下 rep 先切成 concentric 與 eccentric phase，並在 session-level 標準化後使用 PCA 摘要多軸波形變化，再計算 phase range。這類特徵用於描述動作幅度或波形能量是否隨疲勞改變。

第三類是 phase similarity drift。每個 set 以前兩下 rep 作為早期動作模板，計算後續 phase waveform 與 early reps 的相似度變化。若疲勞造成動作控制下降或代償動作增加，後段 rep 可能會逐漸偏離前段穩定模板。

第四類是 angular variation 與 movement rate。陀螺儀變化量可反映關節或肢段旋轉動態，movement rate 則近似描述在單位時間內完成的 phase waveform change。這類特徵用於檢查疲勞是否伴隨動作速度、穩定性或旋轉變異增加。

### 3.3 VO2 特徵設計

VO2 分析使用 set 後不同 lag window：

```text
0, 10, 20, 30, 45, 60 sec
```

主要特徵包含 VO2 mean、VO2 peak、VO2 slope、subject-relative VO2 delta，以及 VO2 與 rep 數的交互項。這樣做的理由是 VO2 不是和單一下 rep 精準同步的 mechanical signal，而是延遲的 physiological-load signal。因此，本研究將 VO2 視為輔助訊號，而不是主要疲勞標籤。

### 3.4 統計分析

由於 Borg/RPE 屬於 ordinal scale，且不同受試者對同一分數的主觀解讀可能不同，本研究使用 Spearman rank correlation 衡量各特徵與 Borg/RPE 的單調關係。Spearman correlation 不假設線性關係，適合用於檢查「特徵越大，RPE 是否傾向越高」這類排序關係。

本研究計算兩種 correlation。

```text
raw Spearman:
直接使用所有受試者與所有動作資料計算特徵和 Borg/RPE 的 rank correlation。

subject+exercise-centered Spearman:
先在每個 subject + exercise group 內扣除該組平均值，再計算 correlation。
```

raw Spearman 用來看整體趨勢；subject+exercise-centered Spearman 用來降低受試者主觀尺度與動作本身差異造成的 confounding。如果某個特徵在 centered 後仍維持相關，表示它比較可能反映同一個人、同一個動作內的疲勞變化，而不只是人與人或動作與動作的 baseline 差異。

## 4. 結果圖設計與論文圖說

建議將第 024 圖放在論文 Results 前半段，作為 fatigue feature relevance analysis。

圖檔：

```text
artifacts_rep_classification/024_imu_fatigue_component_relevance_figure/024_imu_fatigue_component_relevance_summary.png
```

建議圖說：

```text
Fig. X. Feature relevance analysis between IMU-derived movement components, delayed VO2 responses, and Borg/RPE during resistance training. 
(A) Raw Spearman correlations between selected IMU/VO2 components and Borg/RPE. Positive values indicate that the feature increased with Borg/RPE, whereas negative values indicate that the feature decreased as Borg/RPE increased. 
(B) Comparison between raw correlations and within-subject-and-exercise centered correlations. The centered analysis reduces subject-specific RPE scale differences and exercise-dependent baseline effects. 
(C) Exercise-specific heatmap of CE phase-aware IMU feature correlations with Borg/RPE, showing that fatigue-related movement components vary across exercises.
```

中文圖說：

```text
圖 X. 阻力訓練中 IMU 動作成分、延遲 VO2 反應與 Borg/RPE 之特徵關聯分析。
(A) 以 raw Spearman correlation 呈現各 IMU/VO2 成分與 Borg/RPE 的關聯。正值表示該特徵隨 Borg/RPE 上升而增加，負值表示該特徵隨 Borg/RPE 上升而下降。
(B) 比較 raw correlation 與同受試者同動作中心化後的 correlation，用以降低個人主觀 RPE 尺度與不同動作 baseline 差異的影響。
(C) 以動作別 heatmap 呈現 CE phase-aware IMU 特徵與 Borg/RPE 的相關性，顯示疲勞相關動作成分具有動作依賴性。
```

## 5. 結果

### 5.1 圖 A：IMU/VO2 成分與 Borg/RPE 的整體關聯

圖 A 顯示，累積 TUT 是最穩定且最容易解釋的特徵。Accumulated TUT 與 Borg/RPE 的 raw Spearman correlation 為 0.4594，表示在同一組訓練中，隨著累積張力時間增加，受試者回報的 Borg/RPE 通常也會上升。這個結果符合阻力訓練中 fatigue 與 accumulated workload 逐步累積的概念。

延遲 VO2 特徵也出現一定程度的相關。例如 VO2 slope at 45 s 的 raw Spearman correlation 為 0.3639，代表 set 結束後一段時間內 VO2 變化斜率和 Borg/RPE 有關。然而，VO2 mean delta at 10 s 的 raw Spearman correlation 為 -0.3500，方向較不直觀，顯示 raw VO2 容易受到受試者 baseline、休息時間、呼吸延遲與動作種類影響。因此，VO2 在本研究中較適合作為 delayed physiological load covariate，而不是單獨用來預測 rep-level fatigue。

CE phase range 與 Borg/RPE 的 raw Spearman correlation 為 0.3377，phase similarity drift 為 -0.3272。這表示當 RPE 上升時，phase waveform range 傾向增加，而後段 rep 與前段 reps 的相似度傾向下降。這個結果支持 fatigue 可能表現在動作型態變化上，而不只是 rep 時間變長。

Concentric gyro variation 與 phase movement rate 分別達到 0.2830 與 0.2801，屬於中低度相關，但具備明確的 biomechanical 解釋：疲勞可能造成向心期控制變異增加，或使 phase movement dynamics 改變。相較之下，phase timing drift 與 CE ratio drift 較弱，分別為 0.1740 與 0.1011，表示單純依賴「向心時間是否變長」或「CE ratio 是否改變」不足以穩定判斷疲勞。

主要數值如下：

| Component | Feature | Raw Spearman | Centered Spearman | n |
|---|---|---:|---:|---:|
| Accumulated TUT | cumulative active time | 0.4594 | 0.6010 | 1677 |
| Delayed VO2 | VO2 slope at 45 s | 0.3639 | 0.0475 | 96 |
| VO2 baseline delta | VO2 mean delta at 10 s | -0.3500 | -0.2472 | 96 |
| CE phase range | eccentric PCA range mean | 0.3377 | 0.2826 | 143 |
| CE phase similarity | eccentric wave similarity drift | -0.3272 | -0.0433 | 143 |
| Concentric gyro | concentric gyro diff RMS last2 | 0.2830 | 0.1351 | 143 |
| Phase movement rate | eccentric PCA movement rate mean | 0.2801 | 0.3155 | 143 |
| Phase timing drift | concentric duration last2/first2 | 0.1740 | 0.0745 | 143 |
| CE ratio drift | CE ratio slope | 0.1011 | 0.0453 | 143 |

### 5.2 圖 B：raw 與 subject+exercise-centered correlation 比較

圖 B 用來檢查 correlation 是否只是由「不同人主觀尺度不同」或「不同動作本來就比較累」造成。若某個特徵在 centered 後大幅下降，代表它可能較受 baseline confound 影響；若 centered 後仍維持或上升，代表該特徵更可能反映同一個人、同一動作內的疲勞進展。

Accumulated TUT 在 centered 後由 0.4594 上升到 0.6010，是目前最強也最穩定的結果。這表示在同一個 subject/exercise 內，累積 active time 與 RPE 上升具有明顯單調關係。Phase movement rate 在 centered 後也維持 0.3155，表示 movement-rate statistics 可能比單純 timing 更能反映個體內的 fatigue-related movement change。

相對地，VO2 slope at 45 s 從 0.3639 降至 0.0475，phase similarity drift 從 -0.3272 降至 -0.0433。這不表示這些特徵沒有價值，而是表示它們的 raw correlation 可能部分來自動作種類或受試者 baseline 差異。這也提醒後續模型不能只用 pooled data 的 raw correlation 判斷特徵重要性，而應加入 subject calibration 或 mixed-effect / personalized modeling。

### 5.3 圖 C：動作別 CE phase feature heatmap

圖 C 顯示不同 exercise 的疲勞相關特徵並不一致。例如在 shoulder press 與 triceps curl 中，concentric gyroscope variation 的 late-set increase 和 RPE 關係較強；在 biceps curl 中，eccentric waveform similarity decline 和 CE ratio drift 較有訊號；在 one-arm dumbbell row 中，concentric gyro slope、eccentric similarity slope 與 CE ratio slope 皆具有可觀相關。

這個結果對模型設計很重要。它表示 resistance training fatigue 不應用單一規則描述，例如：

```text
concentric duration increases => fatigue
```

較合理的描述應是：

```text
fatigue-related movement change = exercise-specific combination of phase range,
phase similarity, gyroscope variation, movement rate, timing drift, and accumulated TUT.
```

因此，後續模型應採用 exercise-aware phase fatigue score。也就是先辨識動作種類，再依動作類別套用不同的 phase-aware fatigue feature weighting。

## 6. 討論

第 024 圖提供三個主要發現。

第一，IMU 特徵確實包含和 Borg/RPE 相關的訊號，但這些訊號不是單一維度。累積 TUT 最穩定，代表訓練進度與時間負荷是 RPE 的主要驅動因素之一；CE phase range、phase similarity drift、concentric gyro 與 movement rate 則補充了動作型態與控制變化的資訊。這支持 IMU 可作為 fatigue-related movement changes 的量化工具。

第二，單純速度下降或向心時間變長不足以解釋所有疲勞變化。這點和 Zhao 等人在 back squat 中觀察到的結果相呼應：velocity loss 在非爆發式、節奏控制或非力竭情境下不一定能正確反映肌肉疲勞 [3]。本研究目前也看到 phase timing drift 與 CE ratio drift 的 correlation 較弱，因此若只用 TUT 或向心時間建立疲勞模型，可能會漏掉動作控制與波形型態的變化。

第三，疲勞特徵具有動作依賴性。不同訓練動作的主要肌群、關節自由度、手腕 IMU 對動作的敏感方向都不同，因此同一個特徵在不同動作中的有效性不會一致。這與 wearable fatigue detection 文獻中強調 subject-specific 或 task-specific 分析的觀點一致 [5]。因此，後續應使用 exercise-aware 的特徵組合，而不是所有動作共用單一 fatigue rule。

## 7. 論文中可直接使用的段落

### 7.1 Methods 段落

為了驗證 IMU-derived features 是否可作為疲勞相關動作變化的量化指標，本研究在建立自動預測模型前先進行 feature relevance analysis。本分析使用人工標註之 rep boundary 與 concentric/eccentric phase labels，以避免自動切割誤差影響疲勞特徵的判斷。每一下 rep 依 phase 拆分後，計算 phase duration、PCA waveform range、phase movement rate、gyroscope variation、phase waveform similarity to early repetitions，以及 CE ratio 等特徵。Set-level 特徵則以 mean、slope、first-two versus last-two change 與 coefficient of variation 摘要。由於 Borg/RPE 為 ordinal subjective scale，本研究使用 Spearman rank correlation 衡量各特徵與 Borg/RPE 的單調關係。此外，為降低受試者主觀尺度與動作 baseline 差異，本研究同時計算 raw correlation 與 subject+exercise-centered correlation。

### 7.2 Results 段落

Feature relevance analysis showed that accumulated TUT was the most stable correlate of Borg/RPE. The raw Spearman correlation between cumulative active time and Borg/RPE was 0.4594, and increased to 0.6010 after within-subject-and-exercise centering. CE phase-aware features also showed meaningful associations with Borg/RPE, including eccentric PCA range mean (rho = 0.3377), eccentric waveform similarity drift (rho = -0.3272), concentric gyroscope variation in late repetitions (rho = 0.2830), and eccentric PCA movement rate mean (rho = 0.2801). In contrast, phase timing drift and CE ratio drift showed weaker associations, suggesting that fatigue-related movement changes cannot be fully captured by timing features alone. Delayed VO2 features showed moderate raw correlations, but their centered correlations were substantially reduced, indicating that VO2 should be treated as a delayed auxiliary physiological-load feature rather than a direct rep-level fatigue marker.

### 7.3 Discussion 段落

These findings support the use of IMU-derived phase-aware features as indirect markers of fatigue-related movement changes during resistance training. Importantly, the IMU should not be interpreted as directly measuring muscle fatigue; rather, it captures kinematic manifestations that may accompany fatigue, such as increased phase waveform range, decreased similarity to early repetitions, increased gyroscope variation, and altered movement rate. The weak association of CE ratio drift and timing-only features suggests that a fatigue model based solely on repetition duration or concentric duration would be insufficient. The exercise-specific heatmap further indicates that the relevant fatigue-related components vary across exercises, supporting an exercise-aware and phase-aware model design.

### 7.4 研究限制段落

本分析仍有數個限制。第一，Borg/RPE 為主觀量表，不同受試者可能使用不同內部標準，因此跨人比較具有噪音。第二，本分析使用相關係數，因此只能說明特徵與 RPE 的單調關聯，不能推論因果。第三，VO2 資料量較少且具有呼吸延遲，較適合做 set-level delayed physiological load 分析，不適合直接對齊單一下 rep。第四，本分析使用 ground-truth segmentation，代表它驗證的是特徵與 RPE 的理想上限；實際部署時仍需要高品質 rep segmentation 與 CE phase split 才能穩定重現這些特徵。

## 8. 如何在論文中銜接後續模型

第 024 圖後面可以接下列研究架構：

```text
Step 1: rep segmentation
目標是穩定取得每一下 rep 的 start/end。

Step 2: CE phase split
目標是取得 concentric/eccentric phase，讓 TUT、phase range、phase similarity 等特徵可計算。

Step 3: exercise recognition
目標是知道目前是哪個動作，因為不同動作適用的 fatigue features 不同。

Step 4: exercise-aware phase fatigue score
依照動作類別整合 accumulated TUT、phase range、phase similarity、gyro variation、movement rate、VO2 delayed load 與 subject calibration。

Step 5: RPE / training suggestion
輸出 set-level RPE estimate、fatigue trend 或訓練建議。
```

這樣的邏輯會比直接宣稱「IMU 預測肌肉疲勞」更嚴謹，也更容易和文獻對齊。

## 9. 引用文獻與引用理由

[1] T. T. de Beukelaar and D. Mantini, "Monitoring Resistance Training in Real Time with Wearable Technology: Current Applications and Future Directions," Bioengineering, vol. 10, no. 9, p. 1085, 2023, doi: 10.3390/bioengineering10091085.

引用理由：這篇 review 說明穿戴式感測器在阻力訓練中的應用，包括即時監測、生理與生物力學參數，以及 IMU 在阻力訓練研究中的潛力與限制。用來支撐本研究為何使用穿戴式 IMU 監測阻力訓練。

[2] H. Zhao, T. Nishioka, and J. Okada, "Validity of using perceived exertion to assess muscle fatigue during resistance exercises," PeerJ, vol. 10, p. e13019, 2022, doi: 10.7717/peerj.13019.

引用理由：這篇使用 Borg CR-10、sEMG spectral fatigue index 與 velocity measures 評估單關節阻力訓練疲勞，結果指出 RPE 可反映疲勞變化。用來支撐本研究以 Borg/RPE 作為 fatigue-related target 的合理性。

[3] H. Zhao, D. Seo, and J. Okada, "Validity of using perceived exertion to assess muscle fatigue during back squat exercise," BMC Sports Science, Medicine and Rehabilitation, vol. 15, no. 1, p. 14, 2023, doi: 10.1186/s13102-023-00620-8.

引用理由：這篇在 back squat 中比較 RPE 與 velocity loss 作為疲勞指標的有效性，指出 RPE 與 sEMG-based fatigue index 有關，但 velocity loss 在非爆發式情境下不一定能正確反映疲勞。用來支撐本研究不只看速度或時間，而加入 phase waveform 與 gyro variation。

[4] J. J. Gonzalez-Badillo, J. M. Yanez-Garcia, R. Mora-Custodio, and D. Rodriguez-Rosell, "Velocity Loss as a Variable for Monitoring Resistance Exercise," International Journal of Sports Medicine, vol. 38, no. 3, pp. 217-225, 2017, doi: 10.1055/s-0042-120324.

引用理由：這篇是 resistance training 中 velocity loss 監測疲勞與訓練負荷的代表性文獻。用來說明速度下降是常見 fatigue proxy，但本研究結果與後續文獻指出，在非爆發式或節奏控制訓練中不能只依賴 velocity loss。

[5] V. C. H. Chan, S. M. Beaudette, K. B. Smale, K. H. E. Beange, and R. B. Graham, "A Subject-Specific Approach to Detect Fatigue-Related Changes in Spine Motion Using Wearable Sensors," Sensors, vol. 20, no. 9, p. 2646, 2020, doi: 10.3390/s20092646.

引用理由：這篇使用 IMU 與 subject-specific composite index 偵測 fatigue-related kinematic changes，強調疲勞造成的動作變化具有個體差異。用來支撐本研究使用 subject+exercise-centered analysis 與後續 subject calibration 的必要性。

[6] P. Chang, C. Wang, Y. Chen, G. Wang, and A. Lu, "Identification of runner fatigue stages based on inertial sensors and deep learning," Frontiers in Bioengineering and Biotechnology, vol. 11, 2023, doi: 10.3389/fbioe.2023.1302911.

引用理由：這篇使用 IMU time-series 與 deep learning 辨識疲勞階段，並討論 RPE 在 fatigue stage 標記中的應用。雖然任務是跑步而非阻力訓練，但可支撐 IMU 時序訊號包含 fatigue-related movement information。

[7] G. A. V. Borg, "Psychophysical bases of perceived exertion," Medicine and Science in Sports and Exercise, vol. 14, no. 5, pp. 377-381, 1982, doi: 10.1249/00005768-198205000-00012.

引用理由：Borg/RPE 的基礎文獻。用來說明 RPE 是主觀感知強度的量化尺度，因此本研究使用 Spearman rank correlation 而不是只假設連續線性關係。

[8] N. A. Burd, R. J. Andrews, D. W. D. West, J. P. Little, A. J. R. Cochran, A. J. Hector, J. G. A. Cashaback, M. J. Gibala, J. R. Potvin, S. K. Baker, and S. M. Phillips, "Muscle time under tension during resistance exercise stimulates differential muscle protein sub-fractional synthetic responses in men," The Journal of Physiology, vol. 590, no. 2, pp. 351-362, 2012, doi: 10.1113/jphysiol.2011.221200.

引用理由：這篇指出 resistance exercise 中 time under tension 會影響肌肉蛋白合成反應。用來支撐本研究將 cumulative TUT 作為核心負荷與疲勞相關特徵之一。

## 10. 論文中應避免的說法

不建議寫：

```text
IMU directly measures muscle fatigue.
```

建議改寫為：

```text
IMU-derived kinematic features can quantify fatigue-related movement changes associated with Borg/RPE.
```

不建議寫：

```text
Concentric duration increase is the main fatigue marker for all exercises.
```

建議改寫為：

```text
Fatigue-related movement changes were exercise-dependent and were better represented by a combination of accumulated TUT, CE phase waveform range, phase similarity drift, gyroscope variation, and movement-rate features.
```

不建議寫：

```text
VO2 can estimate every repetition's RPE in real time.
```

建議改寫為：

```text
VO2 should be treated as a delayed physiological-load feature and combined with rep-level IMU fatigue states.
```
