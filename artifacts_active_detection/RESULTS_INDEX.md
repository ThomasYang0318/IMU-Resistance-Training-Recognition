# Active / Set Detection Results Index

正式結果資料夾採用遞增編號：

```text
001_<method_or_experiment_name>/
002_<method_or_experiment_name>/
003_<method_or_experiment_name>/
```

低分 threshold 測參數或臨時 exploratory output 不列入正式編號。

## 001_window_rf_action_5fold

目的：

驗證 active / set detection 是否能從 IMU window 特徵泛化到未見過的人。

方法：

- target：`action`
- method：`window-rf`
- split：subject-wise 5-fold GroupKFold
- window：200 samples
- stride：100 samples
- post-process：probability threshold + min segment length + gap merge
- commit：`2205550d`

主要數值：

```text
sample precision: 0.7673
sample recall: 0.9923
sample F1: 0.8654
sample accuracy: 0.7639
segment IoU@0.50 precision: 0.3913
segment IoU@0.50 recall: 0.1324
segment IoU@0.50 F1: 0.1978
true segments: 68
predicted segments: 23
matched segments: 9
mean matched IoU: 0.8170
```

結果檔案：

- `active_detection_metrics.csv`
- `active_detection_metrics_by_subject.csv`
- `active_detection_file_metrics.csv`
- `fold_manifest.csv`
- `summary.json`
- `window_confusion_matrix.csv`
- `window_confusion_matrix.png`
- `window_rf_active_detection_f1.png`
- `timeline_examples/*.png`

解讀：

`window-rf` 能學到 active/rest window-level 特徵，但 set-level IoU 仍低。下一版應改善 boundary-aware post-processing，避免把多組 set 黏在一起或切掉 set 開頭。
