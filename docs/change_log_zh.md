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
