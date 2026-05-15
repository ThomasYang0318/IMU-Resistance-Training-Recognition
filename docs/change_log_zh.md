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
