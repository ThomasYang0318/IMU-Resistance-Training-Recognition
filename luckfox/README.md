# LuckFox Pure-Python Realtime IMU Recognition

This folder contains a deployable realtime recognizer for:

- Bench
- Biceps
- RDL
- Shoulder
- Squat
- Triceps
- Crunch
- Row
- Other

Runtime inference uses only the Python standard library. Training uses `numpy` and `scikit-learn` on the development machine, then exports a JSON forest model that LuckFox can load without external ML packages.

## Files

- `features.py`: Pure-Python causal IMU feature extraction.
- `model.py`: Pure-Python random forest JSON inference.
- `realtime_infer.py`: Device-side realtime recognizer with a ring buffer.
- `simulate_dataset_stream.py`: Replays a dataset CSV sample-by-sample through the realtime recognizer.
- `train_luckfox_model.py`: Development-machine trainer/exporter.

## Train On Development Machine

Example: hold out `yentsen0515workout` and train on the other subjects.

```bash
.venv311/bin/python luckfox/train_luckfox_model.py \
  --data-dir datasets/workout \
  --output-dir luckfox/artifacts/holdout_yentsen \
  --holdout-subject yentsen0515workout \
  --train-window-step 4 \
  --active-estimators 70 \
  --action-estimators 100 \
  --max-depth 16
```

This writes:

- `luckfox/artifacts/holdout_yentsen/luckfox_model.json`
- `luckfox/artifacts/holdout_yentsen/train_summary.json`

## Dataset Stream Test

```bash
.venv311/bin/python luckfox/simulate_dataset_stream.py \
  --model luckfox/artifacts/holdout_yentsen/luckfox_model.json \
  --csv datasets/workout/tsenyu0515workout/tsenyu0515workout_whole_session_20260515_200226.csv \
  --output-dir luckfox/artifacts/holdout_yentsen/stream_test_tsenyu
```

Outputs:

- `stream_predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix_row_proportion.csv`
- `summary.json`

## LuckFox Runtime

On the device, copy:

- `features.py`
- `model.py`
- `realtime_infer.py`
- `luckfox_model.json`

Then feed one CSV line per IMU sample:

```text
timestamp,ax,ay,az,gx,gy,gz,mx,my,mz
```

Run:

```bash
python3 realtime_infer.py --model luckfox_model.json --stdin
```

Output format:

```text
time_seconds,prediction,active_probability,active_threshold,action_candidate,action_confidence,pre_gate_prediction,confirmation_ready,confirmed_action,action_consistency,repetition_peak_count,mad_gate_active,acc_mad,gyro_mad
```

The current model uses a 4 s ring buffer and emits one prediction every 0.5 s after the buffer is filled.

Current realtime decision defaults:

- global active threshold: `0.30`
- RDL active threshold: `0.12`
- active smoothing: 5 emitted windows
- action smoothing: 7 emitted windows
- repetition confirmation gate: enabled
- confirmation window: 4 s
- minimum stable action windows: 8
- minimum action consistency: 0.90
- confirmation streak: 3 emitted windows
- minimum gyro peaks: 2
- MAD activity gate: enabled for newly exported models
- MAD window: 2 s
- MAD rule: `acc_mad >= 0.018 OR gyro_mad >= 1.0`

The lower RDL threshold addresses slow hinge windows that the active detector otherwise classifies as `Other`; action smoothing reduces short shoulder/crunch label flips.
The confirmation gate keeps output as `Other` until the candidate action is stable and repeated-motion-like, which reduces accidental shake false positives at the cost of a few seconds of startup delay.
The MAD gate is a lightweight first-layer activity filter. It is intentionally conservative and should be calibrated per sensor placement before being made stricter.
