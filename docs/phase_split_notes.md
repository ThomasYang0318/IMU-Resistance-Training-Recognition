# 向心 / 離心單刀切割

每個 repetition 已經存成獨立的 `rep*.csv`，所以此 phase-split 任務假設
rep 邊界已經正確，只預測每個 rep 內部唯一一個 `concentric` / `eccentric`
切割點。

目前 baseline 參考常見 wearable resistance-training 的訊號處理做法：

- 在 peak / turning-point 分析前先平滑 IMU trace；
- 動態選擇 dominant movement direction，而不是為每個動作手刻固定軸；
- 將向心 / 離心轉換視為 dominant motion signal 內部的 turning point。

參考文獻：

- Morris et al., RecoFit, CHI 2014：arm-worn inertial sensing 用於 exercise
  segmentation、recognition 與 repetition counting。
- LEAN, Sensors 2023：使用 smoothed inertial data、high-variance / dominant
  direction selection 與 turning points 推估 repetition phase timestamps。
- LiftRight, Smart Health 2020：從 wearable inertial sensing 將 strength-training
  traces 切成 sets、repetitions 與 phases。

基本執行：

```bash
.venv311/bin/python tools/evaluate_phase_split.py --data-dir datasets/workout --session target_session
```

將 `target_session` 換成 session path 的任意子字串。省略 `--session` 時會評估
`datasets/workout` 底下全部 reps。

常用模式：

```bash
# 單刀 baseline：假設 rep 邊界正確，直接切在 rep 內部中點。
# summary 會輸出 sample 誤差、相對 rep 長度誤差與 IoU 指標。
.venv311/bin/python tools/evaluate_phase_split.py --data-dir datasets/workout --method midpoint

# 非監督 IMU turning-point 比較。
.venv311/bin/python tools/evaluate_phase_split.py --data-dir datasets/workout --method signal

# 輸出 waveform split SVG 與 `plot_manifest.csv`。
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --session target_session \
  --method midpoint \
  --output-csv artifacts_phase_split/target_session_summary.csv \
  --write-plots artifacts_phase_split/target_session_waveforms

# 以人為單位切 train / validation。
# 會輸出 `person_split.csv`、每人正確率 / 秒數誤差 / IoU、validation waveform SVG、
# 以及人與人之間的 accuracy / IoU comparison SVG。
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --method supervised-regression \
  --person-split-output artifacts_phase_split/person_split_eval \
  --val-ratio 0.3 \
  --seed 42 \
  --tune-iou-bias

# 新使用者少量標註校正：每個 validation/test 人前 20 筆 rep 作為 calibration，
# 後續 reps 才算真正 test，並輸出校正前 / 校正後比較。
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --method supervised-regression \
  --person-split-output artifacts_phase_split/person_split_eval_personal_calibration \
  --val-ratio 0.3 \
  --seed 42 \
  --tune-iou-bias \
  --personal-calibration-reps 20 \
  --personal-calibration-shrink 0.25

# 指定誰作為 validation/test，其餘所有人作為 training。
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --method supervised-regression \
  --person-split-output artifacts_phase_split/test_yanz \
  --val-people yanz0510workout \
  --seed 42

# 每次留一個人作為 validation/test，也就是 leave-one-person-out k-fold。
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --method supervised-regression \
  --leave-one-person-out-output artifacts_phase_split/leave_one_person_out \
  --no-bias-correction \
  --seed 42

# 使用既有 phase labels 作為 oracle cut，並輸出切割後 phase CSV。
.venv311/bin/python tools/evaluate_phase_split.py \
  --data-dir datasets/workout \
  --session target_session \
  --method phase-column \
  --write-splits artifacts_phase_split/target_session_splits
```
