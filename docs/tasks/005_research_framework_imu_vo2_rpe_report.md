# IMU, VO2 Ground Truth 與 RPE/Borg 關聯之整體研究框架設計

> Task ID: `005_research_framework_imu_vo2_rpe`  
> Branch: `rep-segmentation-classification`  
> Task Owner: `Main Agent`  
> Last Updated: `2026-05-17`

## Abstract

本任務設計一個以腕戴式 IMU、VO2 ground truth 與 Borg/RPE 為核心的阻力訓練研究框架。框架先以 ground-truth rep 與 concentric/eccentric phase 標註排除切割誤差，檢查 IMU-derived TUT、phase waveform、gyro variation 與 set-level trend 是否和 RPE/Borg 及 delayed VO2 具有可解釋關聯，再逐步轉換成 subject-wise、exercise-aware、phase-aware 的 RPE/VO2 融合模型。輸出包含研究問題、假說、資料層級、特徵設計、模型與評估流程、引用對應表，以及一張 paper-ready 研究框架圖。

## Index Terms

IMU, resistance training, Borg RPE, VO2, fatigue-related movement change, time under tension, concentric-eccentric phase, subject-wise validation

## I. Introduction

[S01] 本專題的合理論文定位應是「IMU 量化 fatigue-related movement changes」，而不是宣稱 IMU 直接量測肌肉疲勞本身，因為 IMU 主要提供加速度、角速度與姿態相關的外部動作學訊號 [8], [16], [22]。

[S02] Borg/RPE 可作為阻力訓練中的主觀 exertion label，因為 RPE 源自 perceived exertion 的心理物理量表，且阻力訓練 RPE 的 convergent validity 已由系統性回顧與 meta-analysis 支持 [1], [3]。

[S03] VO2 與 energy expenditure 可作為生理負荷 ground truth 或輔助標籤，但在阻力訓練中應以 set-level delayed response 方式處理，因為呼吸代謝反應不會精準同步於單一下 repetition [13], [14], [19], [20]。

[S04] 因此，本研究框架不直接做「rep-level Borg regression」作為主線，而先建立「phase-aware、exercise-aware、set-level fatigue trend」的證據鏈 [18], [20], [21], [22]。

## II. Task Definition

- Goal: 設計 IMU、VO2 ground truth 與 RPE/Borg 關聯的整體研究框架，並產出 IEEE 風格任務報告與框架圖。
- Inputs: `README.md`, `todo.md`, `proposal.md`, `docs/README.md`, `docs/agent_workflow_zh.md`, `docs/documentation_policy_zh.md`, `docs/artifact_organization_zh.md`, `docs/task_report_template_ieee_zh.md`, `docs/*018/020/022/023/024*`, `artifacts_rep_classification/018/019/021/022/023/024*`, 以及文獻子 Agent 備忘錄。
- Allowed Changes: 新增 `docs/tasks/005_research_framework_imu_vo2_rpe_report.md`；新增 `artifacts/paper_figures/001_imu_vo2_rpe_framework/`；更新 `artifacts/paper_figures/RESULTS_INDEX.md` 與 `docs/README.md` 的短索引。
- Forbidden Changes: 不刪除或回復任何既有 artifacts；不處理 `datasets/raw_data` 的刪除狀態；不修改 schema、核心評估指標、artifact root 或既有論文主敘事。
- Non-goals: 不重跑正式模型；不新增 RAG/MCP；不導入新資料庫；不改 App 或 embedded pipeline。
- Outputs: 本報告、框架圖、引用 traceability CSV、artifact `summary.json` 與 index。
- Acceptance Criteria: 文件需定義研究問題、假說、資料/標籤、特徵、模型、評估、限制；圖需呈現 IMU/RPE/VO2 的資料流與驗證層級；關鍵證據句需在句尾標註來源並列入引用對應表。
- Dependencies: 本任務依賴既有 018/019/021/022/023/024 分析結果，但不依賴重新產生資料。
- Handoff: 下一任務可依本報告開 `006` 實作 `fatigue_rpe_vo2` formal experiment plan 或重跑 subject-wise model。

## III. Input Data and Assumptions

[S05] 既有 018 分析已建立 ground-truth rep/phase 條件下的 Borg/RPE feature dataset，包含 1677 個 merged GT reps 與 6 個 trainable folders，適合作為「排除自動切割誤差後是否存在 RPE 訊號」的上限檢查 [18]。

[S06] 既有 019 VO2 分析已將 VO2 以 0、10、20、30、45、60 秒 lag window 對齊到 set-level，且 random forest 在部分 lag 對 `vo2_mean` 達到中度 Spearman 關聯，支持 VO2 應被建模為 delayed physiological-load signal [19]。

[S07] 既有 022 融合分析顯示，同時具備 RPE 與 VO2 的 overlap 為 96 sets 與 572 lag rows，因此 VO2 融合階段的統計解讀必須明確標示樣本量限制 [20]。

[S08] 既有 023 phase-aware 分析顯示，rep progress、cumulative TUT、phase similarity、phase range、concentric gyro 與 movement-rate features 均含有不同程度 RPE 訊號，但最強訊號仍包含 set/rep progress 與 accumulated workload [21]。

[S09] 既有 024 結果圖將主要成分彙整為 Accumulated TUT、Delayed VO2、VO2 baseline delta、CE phase range、CE phase similarity、Concentric gyro 與 Phase movement rate，這些成分應成為後續模型的核心 feature groups [22]。

## IV. Literature-Guided Research Questions

### A. Research Questions

RQ1: 在 ground-truth rep/CE phase 條件下，哪些 IMU feature groups 與 Borg/RPE 呈現穩定關聯？

RQ2: VO2 ground truth 在阻力訓練中應作為即時 rep label、set-level delayed label，或只作為生理輔助 covariate？

RQ3: subject-wise 與 subject+exercise-centered 分析是否改變 feature relevance 的結論？

RQ4: exercise-aware phase features 是否比單一 timing rule 更能描述疲勞相關動作變化？

RQ5: 當從 ground-truth segmentation 轉到 predicted segmentation 時，哪些指標最能衡量 deployment gap？

### B. Hypotheses

[S10] H1: 累積 TUT、rep progress 與 set progress 會是 RPE/Borg 的穩定基線訊號，因為阻力訓練 RPE 對 workload 與 repetition time 操弄具有 convergent validity，而本地資料也顯示 cumulative TUT 在 subject+exercise-centered 後維持高 Spearman 關聯 [3], [17], [21], [22]。

[S11] H2: Phase-aware waveform features 會提供 timing 以外的訊號，因為 resistance exercise 可被拆成 repetition、concentric/eccentric phase 與 total TUT，而本地 023/024 顯示 phase range、phase similarity drift 與 gyro variation 與 RPE 有關 [7], [8], [21], [22]。

[S12] H3: VO2 應被設計為 delayed physiological load covariate，而不是當下 rep 的 target，因為 VO2/EE 能作為代謝負荷 ground truth，但呼吸代謝量測在阻力訓練中受動作、恢復期與 lag 影響 [13], [14], [19], [20]。

[S13] H4: 模型需採 subject-wise validation 與 subject calibration，因為 RPE 效度受任務與族群影響，HAR/mobile sensor validation 若讓同一受試者資料跨 train/test 會高估泛化，而本地資料也顯示 subject baseline 與 exercise baseline 會影響 raw correlation [2], [15], [20], [22]。

[S14] H5: Exercise-aware weighting 是必要的，因為不同阻力訓練動作有不同 movement dynamics，既有 wearable strength-training intensity recognition 也採先辨識動作再辨識強度的階層式設計，本地 heatmap 亦顯示 CE phase features 的相關方向與大小因 exercise 而異 [6], [22]。

## V. Proposed Framework

### A. Stage 1: Ground-Truth Upper-Bound Analysis

第一階段固定使用人工 rep boundary 與 CE phase label，避免把自動 segmentation 誤差誤判成 feature 無效。輸入為 `018_gt_rep_waveform_borg_dataset.csv`、`019_vo2_set_waveform_dataset.csv` 與 `021/023` set-level feature tables。

核心輸出:

- RPE/Borg feature relevance table。
- VO2 lag relevance table。
- subject+exercise-centered correlation table。
- exercise-feature heatmap。

主要評估:

```text
Spearman rho
MAE for RPE regression
R2 as secondary regression fit
subject-wise / leave-one-subject-out split
subject+exercise-centered Spearman
```

### B. Stage 2: IMU Fatigue State Construction

設計 set-level IMU fatigue state：

```text
IMU fatigue state
= accumulated TUT
  + rep progress
  + phase range
  + phase similarity drift
  + concentric/eccentric gyro variation
  + movement-rate trend
  + exercise-aware feature weights
```

形式化表示：

```text
$z_{s,e,k} = [TUT, progress, range_{CE}, sim_{CE}, gyro_{CE}, rate_{CE}]$

$F_{IMU}(s,e,k) = \alpha_e^\top z_{s,e,k} + b_s$
```

其中 $s$ 是 subject，$e$ 是 exercise，$k$ 是 set 或 rep index，$\alpha_e$ 是 exercise-specific feature weights，$b_s$ 是 subject calibration offset。

### C. Stage 3: VO2 Delayed Load Fusion

VO2 feature 不與單一下 rep 強行對齊，而是在 set 後 lag window 中抽取：

```text
$V_{s,e,k,\ell} = [VO2_{mean}, VO2_{peak}, VO2_{slope}, \Delta VO2_{subject}]_\ell$
```

其中 $\ell \in \{0,10,20,30,45,60\}$ 秒。融合模型可寫成：

```text
$\hat{RPE}_{s,e,k} = f(F_{IMU}(s,e,k), V_{s,e,k,\ell}, subject, exercise)$
```

若資料量不足，先做 nested model comparison：

```text
Model A: metadata + progress
Model B: A + IMU fatigue state
Model C: B + delayed VO2 covariates
Model D: C + subject calibration
```

### D. Stage 4: Deployment Gap Analysis

在 ground-truth framework 確認有效後，再替換為 predicted rep segmentation 與 predicted CE phase split。部署 gap 不只看 RPE MAE，也要同步看：

- rep count error。
- rep IoU@0.5 / IoU@0.75。
- phase boundary MAE。
- TUT MAE。
- feature drift between GT and predicted segmentation。
- final RPE / delta RPE MAE。

## VI. Figure and Table Reading Guide

框架圖輸出：

```text
artifacts/paper_figures/001_imu_vo2_rpe_framework/figures/imu_vo2_rpe_research_framework.png
```

圖的讀法：

- 左側是 input layer，分成 IMU waveform、manual labels、Borg/RPE workbook 與 VO2 metabolic data。
- 中段是 feature layer，先用 GT segmentation 建立上限，再產生 TUT、CE phase、waveform similarity、gyro variation 與 delayed VO2 features。
- 右側是 modeling layer，先做 feature relevance，再做 exercise-aware phase fatigue state，最後融合 delayed VO2 與 subject calibration。
- 底部是 validation layer，包含 subject-wise split、subject+exercise-centered correlation、GT-to-predicted deployment gap 與 limitations。

引用對應表輸出：

```text
artifacts/paper_figures/001_imu_vo2_rpe_framework/tables/citation_traceability.csv
```

## VII. Results

本任務產出一個可直接作為論文 Methods/Study Design 草稿的研究框架，而不是新模型結果。核心結論如下。

[S15] 第一，RPE/Borg 應被視為主觀 exertion label，而非肌肉疲勞本身的直接 ground truth；因此模型目標應優先設定為 RPE trend、delta RPE、set-level final RPE 或 fatigue-related movement score [1], [2], [3], [4], [5]。

[S16] 第二，IMU feature 應先以 ground-truth rep/phase 驗證可學訊號，再逐步替換為 predicted segmentation，因為既有 018/023/024 結果都是在 GT segmentation 條件下建立的 feature upper bound [18], [21], [22]。

[S17] 第三，VO2 ground truth 對本題最適合的角色是 delayed physiological load，而不是即時 RPE label；本地 019/022 已使用 0 到 60 秒 lag windows 並顯示 VO2 correlation 方向會受 baseline 與延遲影響 [19], [20]。

[S18] 第四，後續正式模型不宜只比較 raw pooled correlation，必須同時報告 subject-wise generalization、subject+exercise-centered relation 與少量 subject calibration 的收益 [2], [15], [20], [22]。

### Citation Traceability

| Sentence ID | Evidence Sentence | Source |
|---|---|---|
| S01 | IMU 量化 fatigue-related movement changes，而非直接量測肌肉疲勞。 | [8], [16], [22] |
| S02 | Borg/RPE 可作為阻力訓練主觀 exertion label。 | [1], [3] |
| S03 | VO2/EE 是生理負荷 ground truth 或輔助標籤，但需 delayed set-level 處理。 | [13], [14], [19], [20] |
| S04 | 研究主線應是 phase-aware、exercise-aware、set-level fatigue trend。 | [18], [20], [21], [22] |
| S05 | 018 提供 1677 GT reps 與 6 個 trainable folders 的 RPE 上限檢查。 | [18] |
| S06 | 019 以 0-60 秒 lag 建立 VO2 set-level 分析。 | [19] |
| S07 | 022 RPE+VO2 overlap 為 96 sets 與 572 lag rows。 | [20] |
| S08 | 023 顯示 TUT、phase similarity、phase range、gyro 與 movement-rate features 含 RPE 訊號。 | [21] |
| S09 | 024 彙整主要 IMU/VO2/RPE relevance components。 | [22] |
| S10 | H1: 累積 TUT 與 progress 是穩定 RPE baseline。 | [3], [17], [21], [22] |
| S11 | H2: phase-aware waveform features 提供 timing 外訊號。 | [7], [8], [21], [22] |
| S12 | H3: VO2 是 delayed physiological load covariate。 | [13], [14], [19], [20] |
| S13 | H4: 需要 subject-wise validation 與 calibration。 | [2], [15], [20], [22] |
| S14 | H5: 需要 exercise-aware weighting。 | [6], [22] |
| S15 | RPE/Borg 不是直接肌肉疲勞 ground truth。 | [1], [2], [3], [4], [5] |
| S16 | GT segmentation upper-bound 必須先於 predicted segmentation 部署。 | [18], [21], [22] |
| S17 | VO2 最適合 delayed physiological load 的角色。 | [19], [20] |
| S18 | 評估需包含 subject-wise、centered relation 與 calibration。 | [2], [15], [20], [22] |

## VIII. Limitations

- Borg/RPE 是 ordinal subjective label，不同受試者的同一分數不一定等價。
- VO2 set-level overlap 目前只有 96 sets，不能過度解讀 VO2 融合模型。
- 既有強證據多使用 ground-truth segmentation，部署時會受到 rep/phase prediction error 影響。
- `kg` 不是 relative load，缺少 1RM 或個人最大能力校正。
- 目前研究框架是設計文件，不是正式實驗結果；若要改論文主敘事或核心評估指標，需另行請使用者確認。

## IX. Reproducibility

本任務的驗收指令：

```bash
git status --short
find docs/tasks -maxdepth 1 -type f | sort
find artifacts/paper_figures -maxdepth 3 -type f | sort
rg -n "TBD|UNRESOLVED|待處理" README.md proposal.md todo.md docs
```

驗收時若最後一行只命中各文件中的檢查指令本身，代表沒有新增未決標記。

圖表 artifact：

```text
artifacts/paper_figures/001_imu_vo2_rpe_framework/
  summary.json
  run_config.yaml
  manifest.csv
  tables/citation_traceability.csv
  figures/imu_vo2_rpe_research_framework.png
```

## X. Conclusion

本任務完成一版整體研究框架：先用 ground-truth rep/CE phase 驗證 IMU feature relevance，再將 VO2 設計為 delayed physiological-load covariate，最後以 subject-wise、exercise-aware、phase-aware 的模型評估 RPE/Borg trend。下一步最小任務應是建立 `artifacts/fatigue_rpe_vo2/001_gt_phase_imu_vo2_rpe_framework_eval/` 的正式 experiment plan，先做 nested model comparison，不急著導入新 schema 或大型模型。

## References

[1] G. A. V. Borg, "Psychophysical bases of perceived exertion," *Medicine & Science in Sports & Exercise*, vol. 14, no. 5, pp. 377-381, 1982, doi: [10.1249/00005768-198205000-00012](https://doi.org/10.1249/00005768-198205000-00012).

[2] M. J. Chen, X. Fan, and S. T. Moe, "Criterion-related validity of the Borg ratings of perceived exertion scale in healthy individuals: A meta-analysis," *Journal of Sports Sciences*, vol. 20, no. 11, pp. 873-899, 2002, doi: [10.1080/026404102320761787](https://doi.org/10.1080/026404102320761787).

[3] J. W. D. Lea, J. M. O'Driscoll, S. Hulbert, J. Scales, and J. Wiles, "Convergent validity of ratings of perceived exertion during resistance exercise in healthy participants: A systematic review and meta-analysis," *Sports Medicine - Open*, vol. 8, no. 1, article 2, 2022, doi: [10.1186/s40798-021-00386-8](https://doi.org/10.1186/s40798-021-00386-8).

[4] H. Zhao, T. Nishioka, and J. Okada, "Validity of using perceived exertion to assess muscle fatigue during resistance exercises," *PeerJ*, vol. 10, e13019, 2022, doi: [10.7717/peerj.13019](https://doi.org/10.7717/peerj.13019).

[5] H. Zhao, D. Seo, and J. Okada, "Validity of using perceived exertion to assess muscle fatigue during back squat exercise," *BMC Sports Science, Medicine and Rehabilitation*, vol. 15, article 14, 2023, doi: [10.1186/s13102-023-00620-8](https://doi.org/10.1186/s13102-023-00620-8).

[6] I. Pernek, G. Kurillo, G. Stiglic, and R. Bajcsy, "Recognizing the intensity of strength training exercises with wearable sensors," *Journal of Biomedical Informatics*, vol. 58, pp. 145-155, 2015, doi: [10.1016/j.jbi.2015.09.020](https://doi.org/10.1016/j.jbi.2015.09.020).

[7] C. Viecelli, D. Graf, D. Aguayo, E. Hafen, and R. M. Fuechslin, "Using smartphone accelerometer data to obtain scientific mechanical-biological descriptors of resistance exercise training," *PLOS ONE*, vol. 15, no. 7, e0235156, 2020, doi: [10.1371/journal.pone.0235156](https://doi.org/10.1371/journal.pone.0235156).

[8] T. T. de Beukelaar and D. Mantini, "Monitoring resistance training in real time with wearable technology: Current applications and future directions," *Bioengineering*, vol. 10, no. 9, article 1085, 2023, doi: [10.3390/bioengineering10091085](https://doi.org/10.3390/bioengineering10091085).

[9] L. Sanchez-Medina and J. J. Gonzalez-Badillo, "Velocity loss as an indicator of neuromuscular fatigue during resistance training," *Medicine & Science in Sports & Exercise*, vol. 43, no. 9, pp. 1725-1734, 2011, doi: [10.1249/MSS.0b013e318213f880](https://doi.org/10.1249/MSS.0b013e318213f880).

[10] F. M. Clemente, Z. Akyildiz, J. Pino-Ortega, and M. Rico-Gonzalez, "Validity and reliability of the inertial measurement unit for barbell velocity assessments: A systematic review," *Sensors*, vol. 21, no. 7, article 2511, 2021, doi: [10.3390/s21072511](https://doi.org/10.3390/s21072511).

[11] J. Staudenmayer, D. Pober, S. Crouter, D. Bassett, and P. Freedson, "An artificial neural network to estimate physical activity energy expenditure and identify physical activity type from an accelerometer," *Journal of Applied Physiology*, vol. 107, no. 4, pp. 1300-1307, 2009, doi: [10.1152/japplphysiol.00465.2009](https://doi.org/10.1152/japplphysiol.00465.2009).

[12] S. L. Kozey, K. Lyden, C. A. Howe, J. W. Staudenmayer, and P. S. Freedson, "Accelerometer output and MET values of common physical activities," *Medicine & Science in Sports & Exercise*, vol. 42, no. 9, pp. 1776-1784, 2010, doi: [10.1249/MSS.0b013e3181d479f2](https://doi.org/10.1249/MSS.0b013e3181d479f2).

[13] G. A. Joao *et al*., "Acute behavior of oxygen consumption, lactate concentrations, and energy expenditure during resistance training: Comparisons among three intensities," *Frontiers in Sports and Active Living*, vol. 3, article 797604, 2021, doi: [10.3389/fspor.2021.797604](https://doi.org/10.3389/fspor.2021.797604).

[14] A. J. Cook *et al*., "Instantaneous VO2 from a wearable device," *Medical Engineering & Physics*, vol. 52, pp. 41-48, 2018, doi: [10.1016/j.medengphy.2017.12.008](https://doi.org/10.1016/j.medengphy.2017.12.008).

[15] H. Braganca, J. G. Colonna, H. A. B. F. Oliveira, and E. Souto, "How validation methodology influences human activity recognition mobile systems," *Sensors*, vol. 22, no. 6, article 2360, 2022, doi: [10.3390/s22062360](https://doi.org/10.3390/s22062360).

[16] V. C. H. Chan, S. M. Beaudette, K. B. Smale, K. H. E. Beange, and R. B. Graham, "A subject-specific approach to detect fatigue-related changes in spine motion using wearable sensors," *Sensors*, vol. 20, no. 9, article 2646, 2020, doi: [10.3390/s20092646](https://doi.org/10.3390/s20092646).

[17] N. A. Burd *et al*., "Muscle time under tension during resistance exercise stimulates differential muscle protein sub-fractional synthetic responses in men," *The Journal of Physiology*, vol. 590, no. 2, pp. 351-362, 2012, doi: [10.1113/jphysiol.2011.221200](https://doi.org/10.1113/jphysiol.2011.221200).

[18] Project artifact, "018 Borg / REP 與 GT 波形特徵關聯上限測試," `docs/borg_waveform_relation_018_zh.md` and `artifacts_rep_classification/018_borg_gt_waveform_relation/summary.json`, 2026-05-17.

[19] Project artifact, "019 VO2 GT waveform relation," `artifacts_rep_classification/019_vo2_gt_waveform_relation/summary.json`, 2026-05-17.

[20] Project artifact, "022 即時 RPE 特徵與 VO2 融合分析," `docs/realtime_rpe_vo2_features_022_zh.md` and `artifacts_rep_classification/022_realtime_rpe_vo2_feature_correlation/summary.json`, 2026-05-17.

[21] Project artifact, "023 CE Phase-Aware Fatigue 與 RPE 驗證," `docs/phase_aware_fatigue_ce_rpe_023_zh.md` and `artifacts_rep_classification/023_phase_aware_fatigue_ce_rpe_analysis/summary.json`, 2026-05-17.

[22] Project artifact, "024 IMU 疲勞相關成分結果圖," `docs/imu_fatigue_component_relevance_024_paper_zh.md` and `artifacts_rep_classification/024_imu_fatigue_component_relevance_figure/summary.json`, 2026-05-17.

[23] Project index, "Rep / Phase / Fatigue Results Index," `artifacts_rep_classification/RESULTS_INDEX.md`, 2026-05-17.
