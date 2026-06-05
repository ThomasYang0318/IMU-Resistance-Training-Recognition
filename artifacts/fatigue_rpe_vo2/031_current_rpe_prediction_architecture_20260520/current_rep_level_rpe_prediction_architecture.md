# Current Rep-Level RPE Prediction Architecture

Input:
- 9-axis IMU raw waveform
- Ground-truth rep and CE phase segments
- Rep-level RPE workbook labels

Alignment:
- Build one row per completed rep
- Exclude unfinished reps
- Attach subject, exercise, set, rep identifiers

Features:
- Exercise one-hot
- Set index
- Rep progress
- Cumulative TUT
- Load kg
- Evaluated but not selected as final main model: phase duration, movement rate, PCA/gyro/similarity features

Model:
- Ridge regression
- Median imputation
- Standardization
- Alpha selected on validation subject

Validation:
- Subject-disjoint 7-fold rotation
- 5 train subjects / 1 validation subject / 1 held-out test subject

Output:
- Continuous RPE 1-10
- Rounded RPE class for confusion matrix
- MAE, Spearman, +/-1 accuracy
