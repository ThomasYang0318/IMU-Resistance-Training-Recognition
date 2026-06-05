# 024 IMU 疲勞相關成分結果圖

## 目的

使用結果圖支撐研究開頭的論述：

> IMU 不能直接量測肌肉疲勞，但可以量化疲勞相關的動作學變化。

本版參考感測器特徵分析常用的呈現方式：以 `Spearman correlation bar chart` 顯示各成分與 Borg/RPE 的關聯度，並以 `exercise-feature heatmap` 顯示不同動作的 CE phase 特徵差異。

## 輸出

```text
artifacts_rep_classification/024_imu_fatigue_component_relevance_figure/
```

論文撰寫版：

```text
docs/imu_fatigue_component_relevance_024_paper_zh.md
```

主要檔案：

- `024_imu_fatigue_component_relevance_summary.png`
- `024_imu_fatigue_component_bar.png`
- `024_imu_fatigue_component_relevance_table.csv`
- `024_exercise_phase_feature_heatmap_values.csv`

## 主圖說明

主圖分成三塊：

1. IMU / VO2 成分與 Borg/RPE 的 Spearman 相關度；
2. raw correlation 與同人同動作校正後 correlation 比較；
3. 每個動作的 CE phase fatigue feature heatmap。

## 主要數值

```text
Accumulated TUT       rho =  0.4594
Delayed VO2 slope     rho =  0.3639
VO2 baseline delta    rho = -0.3500
CE phase range        rho =  0.3377
CE phase similarity   rho = -0.3272
Concentric gyro       rho =  0.2830
Phase movement rate   rho =  0.2801
Phase timing drift    rho =  0.1740
CE ratio drift        rho =  0.1011
```

## 解讀

這張圖支持以下說法：

```text
IMU 可量化疲勞相關的動作學表徵，包括累積 TUT、CE phase waveform range、phase similarity drift、concentric gyroscope variation、phase movement rate 與 phase timing drift。
```

但圖中也顯示：

```text
CE ratio drift 與單純向心時間變長不是最強訊號。
```

因此論文不應寫成「IMU 可直接量測肌肉疲勞」，而應寫成：

```text
IMU 可用於估計 fatigue-related movement changes。
```

## 建議放在論文中的句子

```text
To justify the use of IMU-derived features as fatigue-related movement indicators, we first quantified the association between Borg/RPE and multiple IMU components. Spearman correlation analysis showed that accumulated TUT, CE phase waveform range, phase similarity drift, concentric gyroscope variation, and phase movement-rate features were associated with perceived exertion, supporting the use of phase-aware IMU features as indirect markers of fatigue-related movement changes.
```

中文：

```text
為了驗證 IMU 特徵是否可作為疲勞相關動作表徵，本研究先量化 Borg/RPE 與多種 IMU 成分之關聯。Spearman 相關分析顯示，累積 TUT、CE phase 波形範圍、phase 相似度漂移、向心期陀螺儀變化與 phase movement rate 皆和主觀疲勞程度呈現關聯，支持 phase-aware IMU features 可作為疲勞相關動作變化的間接量化指標。
```

## 限制

- Borg/RPE 是主觀尺度，不是肌肉疲勞的直接生理標籤；
- 相關不代表因果；
- VO2 是延遲生理負荷，不能直接對齊單一下 rep；
- 不同動作的疲勞特徵不同，因此後續模型應採 exercise-aware 設計。
