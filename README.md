# XTinyHAR Resistance Student (Multi-File)

This project trains a first-stage inertial student model for **action classification only** using CSV IMU data.

## Expected Input Schema

Required CSV columns:

- `sensor_ts`
- `ax`, `ay`, `az`, `gx`, `gy`, `gz`
- `action_type`
- `subject_id`

## Project Layout

- `datasets/custom_resistance_dataset.py`: CSV loading, schema checks, sequence assembly
- `preprocessing/window_pipeline.py`: resampling, z-score, subject-wise split, sliding windows
- `models/inertial_student.py`: XTinyHAR-style student model
- `train/train_student.py`: training entrypoint (subject-wise split only)
- `deploy/export_onnx.py`: checkpoint to ONNX export
- `deploy/luckfox_infer.py`: ONNX runtime sliding-window inference helper

## Python Version

- Python `3.10+`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Edit `config.yaml`:

- `data.data_dir` and `data.csv_glob`
- training/device settings
- window/model parameters

## Train

```bash
python -m train.train_student --config config.yaml
```

Artifacts are written to `io.output_dir` (default: `./artifacts_student`):

- `student_best.pt`
- `label_map.json`
- `zscore_stats.json`
- `train_summary.json`
- `effective_config.json`

## Export ONNX

```bash
python -m deploy.export_onnx --config config.yaml
```

Optional overrides:

```bash
python -m deploy.export_onnx --config config.yaml --checkpoint ./artifacts_student/student_best.pt --output ./artifacts_student/student_model.onnx --opset 17
```

## Luckfox-Style ONNX Inference

```bash
python -m deploy.luckfox_infer \
  --onnx ./artifacts_student/student_model.onnx \
  --stats ./artifacts_student/zscore_stats.json \
  --label-map ./artifacts_student/label_map.json \
  --csv ./your_stream_or_record.csv
```

This script uses a rolling window buffer and prints one predicted action label per ready window.
