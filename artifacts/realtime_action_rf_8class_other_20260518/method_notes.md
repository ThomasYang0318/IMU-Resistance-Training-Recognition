# Realtime 8-class + Other action recognition

## Online model

- Input: 9-axis IMU (`ax ay az gx gy gz mx my mz`) from whole-session CSV files.
- Windowing: causal 2.00s window, updated every 0.50s.
- Labels: active `concentric/eccentric` samples become one of the 8 exercise labels; all rest, `none`, transition, or ambiguous windows become `Other`.
- Features: per-axis mean, standard deviation, min/max/range, RMS energy, median, mean absolute value, last-first delta, derivative variability, 3-axis magnitudes, and within-sensor axis correlations.
- Classifier: Random Forest, trained with subject-wise GroupKFold. Training folds downsample `Other` to 1.00x active-window count; test folds keep all stream windows.
- Final online output: causal majority smoothing over the current and previous 10 predictions (`k=11`, 5.5s history at 0.5s stride). This uses no future windows, so it is still online, but it adds output inertia around action transitions.

## Labels

- db_bench_press
- db_biceps_curl
- db_rdl
- db_shoulder_press
- db_squat
- db_triceps_curl
- db_weighted_crunch
- one_arm_db_row
- Other

## References used

- Bao, L. and Intille, S. S. (2004). *Activity Recognition from User-Annotated Acceleration Data*. Used for sliding-window wearable acceleration features such as mean, energy, entropy/correlation style descriptors. https://www.ccs.neu.edu/home/intille/papers-files/BaoIntille04.pdf
- Breiman, L. (2001). *Random Forests*. Used for the ensemble tree classifier. https://doi.org/10.1023/A:1010933404324
