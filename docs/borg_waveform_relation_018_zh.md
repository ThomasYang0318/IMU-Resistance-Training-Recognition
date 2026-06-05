# 018 Borg / REP 與 GT 波形特徵關聯上限測試

> 2026-05-17 artifact cleanup note：`018_borg_gt_waveform_relation_exclude_sparse/` 已瘦身刪除；目前保留 `018_borg_gt_waveform_relation/` 作為 RPE/Borg 主線上限分析。

## 目的

這版不是測自動切割，而是先回答一個更前面的問題：

> 如果使用 ground-truth rep / phase 切割點，原始 IMU 波形、波形相似度、TUT、向心 / 離心時間是否能預測每一下的 Borg/RPE 值？

如果連 ground-truth 切割都無法從波形學到 Borg/RPE 關聯，那後續再改自動 rep segmentation 對訓練建議的價值就有限。

## Borg 表解析規則

每個 subject 資料夾中若有同名 `.xlsx`，會讀取第一個工作表：

- row key：`1_0`、`1_1` 這類格式；
  - 前面的數字 `1..8` 對應當天動作順序；
  - 後面的數字對應 CSV 裡的 set id；
- 欄位 `0..11`：rep index；
- cell value：該 rep 的 Borg/RPE；
- 空白：沿用前一個 rep 的 Borg/RPE；
- `X`：該 rep 沒做完，不納入 Borg target training；
- `kg` / `KG`：該 set 的重量。

`thomas0506workout` 沒有同名 Borg/RPE workbook，因此不放入 training。它只能作為沒有 Borg target 的外部 waveform 展示/驗證，不能計算 Borg prediction accuracy。

## 特徵

這版只用 ground-truth labels 切 rep，不使用 predicted segmentation。

每一下 rep 抽：

- metadata：rep index、kg、exercise one-hot；
- TUT：rep duration、concentric duration、eccentric duration、phase ratio；
- waveform statistics：9 軸與 acc/gyro/mag magnitude 的 mean/std/range/RMS/diff/slope；
- PCA waveform：9-axis PCA principal motion；
- waveform similarity：該 rep 與同 set 第一 rep、前一 rep 的 cosine similarity。

## 正式結果

輸出：

```text
artifacts_rep_classification/018_borg_gt_waveform_relation/
```

資料狀態：

```text
raw Borg target reps = 1425
completed Borg target reps = 1416
merged GT waveform reps = 1408
```

有完整 Borg/RPE target 的主要 folders：

```text
haoyu0512workout
hsianshun0514workout
tsenyu0515workout
yanz0510workout
yoru0511workout
```

`yushuan0513workout` 只有 12 個 target 且全部 Borg = 1，因此另外跑了排除稀疏標註的 sensitivity。

## Cross-subject 上限測試結果

使用 ground-truth rep/phase 切割，leave-subject-out cross validation。

### 含 yushuan sparse target

```text
global mean baseline MAE = 1.7274
exercise mean baseline MAE = 1.6489
metadata RF MAE = 1.6440
TUT RF MAE = 1.7033
waveform RF MAE = 1.6327
combined RF MAE = 1.6390
```

### 排除 yushuan sparse target

輸出：

```text
artifacts_rep_classification/018_borg_gt_waveform_relation_exclude_sparse/
```

```text
global mean baseline MAE = 1.7071
exercise mean baseline MAE = 1.6297
metadata RF MAE = 1.6228
TUT RF MAE = 1.6832
waveform RF MAE = 1.5554
combined RF MAE = 1.5741
```

最佳是 `waveform RF`：

```text
MAE = 1.5554 Borg
R2 = 0.1829
Spearman = 0.4394
rounded +/-1 accuracy = 0.5387
```

## 解讀

波形確實含有一些 Borg/RPE 訊號，但目前關聯不強。

關鍵觀察：

1. 單純 metadata / exercise / rep index 已經能接近 waveform 結果，表示 Borg 大部分仍由動作、重量、rep 位置與個人主觀尺度影響。
2. TUT-only 沒有比 metadata 好，代表只看向心/離心時間不足以預測 Borg。
3. waveform RF 比 baseline 好一些，代表原始波形變化與疲勞感有訊號，但還不夠強到可直接做高可信 Borg 估計。
4. 目前是 cross-subject 預測，Borg/RPE 是主觀量表，跨人差異會壓低泛化；few-shot per-subject calibration 可能很重要。

## 對 Rep Segmentation 的意義

這個結果不代表 rep segmentation 沒用，但代表要避免過度期待：

- 若目標只是預測 Borg/RPE，單靠自動切 rep + TUT 不夠；
- 若目標是訓練建議，應把 rep segmentation 當作特徵來源之一，而不是唯一訊號；
- 更有價值的方向是「同一個人的 set 內趨勢」：
  - waveform similarity 是否逐 rep 下降；
  - TUT 是否變長或變短；
  - concentric/eccentric ratio 是否改變；
  - amplitude / velocity proxy 是否下降。

## 下一步

1. 改成 within-subject / few-shot Borg calibration：
   - 用前幾組有 Borg 的資料校正個人主觀尺度；
   - 再預測同一人的後續 sets。
2. 從絕對 Borg 改成預測 Borg change：
   - `delta Borg from rep 1`
   - `last rep Borg`
   - `set-level Borg slope`
3. 加入 set-level 特徵：
   - 每組 waveform similarity decline；
   - rep duration slope；
   - concentric/eccentric ratio slope；
   - amplitude / velocity proxy decline。
4. 先用 ground-truth segmentation 做完上限，確認有效後，再替換成 predicted segmentation。
