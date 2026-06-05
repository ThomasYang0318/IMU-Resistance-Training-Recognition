# 智慧穿戴阻力訓練成效評估系統 Proposal

最後更新：2026-05-17

## 1. 專題目標

本專題要建立一套以腕戴式 IMU 為核心的阻力訓練成效評估系統，從連續運動資料中辨識動作、切割每一下 repetition、切分向心與離心階段，並計算訓練量、動作品質與疲勞相關特徵。短期目標是完成可重複驗證的離線分析 pipeline；中期目標是建立可部署到 App 與嵌入式裝置的即時推論架構。

## 2. 使用情境

- 使用者配戴腕戴式 IMU 進行啞鈴或徒手阻力訓練。
- 系統接收 6 軸或 9 軸 IMU 時序資料。
- 系統輸出每組 set 的動作種類、次數、每一下起訖、向心/離心時間、TUT、節奏穩定度與疲勞趨勢。
- 未來 App 顯示訓練紀錄、即時提示與每組品質摘要。

## 3. MVP 範圍

第一階段 MVP 聚焦「離線、可驗證、可重跑」：

1. 讀取現有 `datasets/workout` 或 raw CSV。
2. 統一 IMU 欄位、取樣率、時間軸與標註格式。
3. 偵測 active / rest 與 set 邊界。
4. 對 active set 切割 repetition。
5. 對每一下切分 concentric / eccentric phase。
6. 計算 TUT、tempo、ROM proxy、速度/能量 proxy、穩定度與疲勞 proxy。
7. 以 subject-wise split 評估，避免同一受試者同時出現在 train/test。
8. 產出 CSV、JSON、圖表與文件化結果。

暫不納入第一階段 MVP：

- App UI 完整實作。
- 嵌入式即時韌體。
- 雲端帳號、同步與多裝置管理。
- 醫療等級診斷結論。

## 4. 資料輸入與標註

### 4.1 原始輸入

預期每筆 IMU CSV 至少包含：

- `sensor_ts` 或等價時間欄位。
- 加速度：`ax`, `ay`, `az`。
- 陀螺儀：`gx`, `gy`, `gz`。
- 若可用，磁力計或姿態相關欄位可作為 9 軸延伸。
- `subject_id`。
- `action_type`。
- 若已有人工標註：`set_id`, `rep_id`, `phase`。

### 4.2 標註層級

資料標註分成四層：

1. Session：一次完整訓練紀錄。
2. Set：一組連續執行的動作。
3. Rep：一次完整 repetition。
4. Phase：每一下中的 `concentric`、`eccentric`，必要時保留 `transition` 或 `unlabeled`。

## 5. 資料流程

```text
Raw IMU CSV
  -> Data validation
  -> Resampling / filtering / normalization
  -> Active-rest detection
  -> Set boundary detection
  -> Exercise recognition
  -> Repetition segmentation
  -> Phase segmentation
  -> Feature extraction
  -> Quality / fatigue scoring
  -> Evaluation reports
  -> Export for App / embedded inference
```

各階段輸入輸出：

| 階段 | 輸入 | 輸出 | 主要驗收 |
| --- | --- | --- | --- |
| 資料驗證 | CSV | schema report | 欄位、時間軸、缺失值可被檢查 |
| 前處理 | raw samples | clean samples | 取樣率一致、訊號可重現 |
| Active/set 偵測 | clean samples | set segments | set IoU、boundary error |
| 動作辨識 | window 或 set | exercise label | subject-wise macro F1 |
| Rep 切割 | set samples | rep segments | rep IoU、count error |
| Phase 切分 | rep samples | phase segments | phase boundary error、TUT error |
| 特徵萃取 | rep/phase segments | feature table | feature 定義可測試 |
| 評估輸出 | predictions + labels | metrics/artifacts | 可重跑、可比較 |

## 6. 模組切分

建議逐步整理成以下模組。現有 `tools/` 先保留作為實驗入口，穩定邏輯再下沉到 package 模組。

```text
imu_project/
  io/
    schemas.py            # CSV 欄位、資料型別與標註 schema
    loaders.py            # 讀取 raw/workout datasets
    writers.py            # 輸出 segments、features、metrics
  preprocessing/
    resampling.py         # 取樣率統一
    filtering.py          # 平滑、低通、高通、重力處理
    normalization.py      # subject/session normalization
  segmentation/
    active_set.py         # active/rest 與 set boundary
    reps.py               # repetition segmentation
    phases.py             # concentric/eccentric segmentation
    postprocess.py        # gap merge、最短長度、邊界修正
  recognition/
    window_classifier.py  # window-level action recognition
    sequence_model.py     # TCN/MS-TCN 等 sample-wise model
  features/
    temporal.py           # duration、tempo、TUT
    waveform.py           # amplitude、smoothness、energy
    fatigue.py            # set/rep 趨勢與 fatigue proxy
    quality.py            # 動作品質分數
  evaluation/
    splits.py             # subject-wise split
    metrics.py            # IoU、F1、count error、TUT error
    reports.py            # 表格與圖表輸出
  deploy/
    onnx_export.py        # model export
    realtime_pipeline.py  # streaming inference prototype
```

短期不需要一次搬完現有程式。每次新增穩定功能時，才把可測試的純邏輯放進模組，`tools/` 只負責 CLI 與組合流程。

## 7. 演算法策略

### 7.1 Baseline 優先

每個核心問題先建立可解釋 baseline：

- Active/set：energy envelope、hysteresis、gap merge。
- Rep：dominant axis peaks、PCA axis、autocorrelation、candidate boundary scoring。
- Phase：rep 內主運動軸速度/角速度 extrema、零交越、能量轉折。
- Quality：節奏穩定度、左右/前後晃動 proxy、range proxy、jerk/smoothness。
- Fatigue：rep duration trend、peak velocity decay、amplitude decay、TUT accumulation、intra-set variability。

### 7.2 ML / DL 擴充

在 baseline 具備可重跑評估後，再加入：

- Window classifier for action recognition。
- Sequence labeling model for sample-wise macro/micro phase。
- Lightweight model for embedded deployment。

模型驗證一律使用 subject-wise split，避免資料洩漏。

## 8. 文件與實驗產物規範

根目錄文件固定分工：

- `README.md`：專案入口、目前路線、執行指令與重要文件索引。
- `proposal.md`：系統目標、資料流程、模組切分、評估方法與開發規範。
- `todo.md`：可執行任務、優先順序、預期輸出與驗收。

`docs/` 固定放方法說明、實驗紀錄、文獻比較、結果解讀與規範文件。文件索引維護在 `docs/README.md`。

正式實驗產物使用：

```text
artifacts/<domain>/<experiment_id>_<short_slug>/
```

每個正式實驗至少包含：

- `summary.json`
- 可機器讀取的 metrics 或 table CSV。
- 必要圖表放在 `figures/`。
- 診斷案例放在 `diagnostics/`。
- 對應 artifact root 的 `RESULTS_INDEX.md` 紀錄。

既有 artifacts 已依論文敘事鏈瘦身；從下一個正式實驗開始採用新分類 root。完整規範見 `docs/artifact_organization_zh.md`。

多 Agent 任務需依 `docs/agent_workflow_zh.md` 派工；文件維護需依 `docs/documentation_policy_zh.md` 控制長度。短期不導入完整 RAG 或額外 MCP，除非觸發條件達成並經使用者決策。

## 9. 評估指標

| 任務 | 指標 |
| --- | --- |
| Active/set detection | segment IoU、boundary MAE、precision/recall/F1 |
| Exercise recognition | accuracy、macro F1、confusion matrix |
| Rep segmentation | count error、rep IoU@0.5、boundary MAE |
| Phase segmentation | phase boundary MAE、concentric/eccentric duration error |
| TUT | per-rep TUT MAE、per-set TUT MAE |
| Quality/fatigue features | correlation with RPE/Borg/VO2 if available, intra-set trend stability |
| Deployment | latency、memory、model size、streaming delay |

## 10. 開發規範

每次小任務開始前要確認：

- 目標：這次要解決哪一個最小問題。
- 輸入：使用哪些檔案、資料表或函式。
- 輸出：新增或修改哪些檔案與產物。
- 限制：不處理哪些範圍、不可破壞哪些既有結果。
- 驗收標準：要跑哪些測試、指令或檢查。

每次功能實作要求：

- 程式修改前先列出會改的檔案。
- 可測試邏輯要有單元測試。
- CLI 或實驗流程要提供執行指令。
- 更新 `proposal.md`、`todo.md`、`README.md` 中相關內容。
- 不覆蓋既有 artifacts，新的實驗輸出使用遞增版本或明確 output dir。

## 11. 近期里程碑

1. 文件與任務架構整理。
2. 建立 core package skeleton 與測試框架。
3. 把 schema validation 與資料讀取從實驗 script 中抽出。
4. 建立 repetition / phase segment 的共用資料結構。
5. 建立 feature extraction 的第一版可測試函式。
6. 整理一鍵 pipeline：raw -> set -> rep -> phase -> features -> report。
7. 準備 App / embedded 需要的輸出介面。
