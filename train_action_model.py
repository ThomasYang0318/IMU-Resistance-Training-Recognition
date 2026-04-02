# train_action_model.py
# 需求:
# pip install numpy pandas scikit-learn joblib

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# =========================
# 可調參數
# =========================
DATA_GLOB = "./dataset/*.csv"

# 你的六軸欄位
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

# 動作標籤欄位
LABEL_COL = "action"

# 如果有 subject_id，建議開 True 做跨受試者切分
USE_GROUP_SPLIT = True
GROUP_COL = "subject_id"   # 若 CSV 沒這欄，程式會自動 fallback 成一般切分

# 取樣率假設 50 Hz；2 秒 window = 100 samples
WINDOW_SIZE = 100
STEP_SIZE = 50

RANDOM_STATE = 42

MODEL_PATH = "action_rf_model.pkl"
ENCODER_PATH = "action_label_encoder.pkl"
META_PATH = "action_model_meta.json"

# 八類標籤映射，方便你統一資料名稱
VALID_ACTIONS = {
    "db_bench_press": "啞鈴臥推",
    "one_arm_db_row": "單手啞鈴划船",
    "db_shoulder_press": "啞鈴肩推",
    "db_biceps_curl": "啞鈴二頭彎舉",
    "db_triceps_extension": "啞鈴三頭彎舉",
    "db_squat": "啞鈴深蹲",
    "db_rdl": "啞鈴羅馬尼亞硬舉",
    "weighted_crunch": "啞鈴負重卷腹",
}


# =========================
# Feature Engineering
# 主流 baseline: time-domain statistics + magnitude
# =========================
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def zero_crossing_rate(x: np.ndarray) -> float:
    # 對 centered signal 算 zero-crossings
    xc = x - np.mean(x)
    return float(np.sum(np.diff(np.signbit(xc)).astype(np.int32))) / max(len(x) - 1, 1)


def signal_energy(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)))


def extract_features(window_df: pd.DataFrame) -> list[float]:
    feats = []

    # 每軸 time-domain features
    for col in SENSOR_COLS:
        x = window_df[col].to_numpy(dtype=np.float32)

        feats.extend([
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
            float(np.max(x) - np.min(x)),
            rms(x),
            signal_energy(x),
            zero_crossing_rate(x),
        ])

    # magnitude features
    acc_mag = np.sqrt(
        np.square(window_df["ax"].to_numpy(dtype=np.float32)) +
        np.square(window_df["ay"].to_numpy(dtype=np.float32)) +
        np.square(window_df["az"].to_numpy(dtype=np.float32))
    )

    gyro_mag = np.sqrt(
        np.square(window_df["gx"].to_numpy(dtype=np.float32)) +
        np.square(window_df["gy"].to_numpy(dtype=np.float32)) +
        np.square(window_df["gz"].to_numpy(dtype=np.float32))
    )

    for mag in [acc_mag, gyro_mag]:
        feats.extend([
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.min(mag)),
            float(np.max(mag)),
            float(np.max(mag) - np.min(mag)),
            rms(mag),
            signal_energy(mag),
            zero_crossing_rate(mag),
        ])

    # 軸間相關性
    acc = window_df[["ax", "ay", "az"]].to_numpy(dtype=np.float32)
    gyr = window_df[["gx", "gy", "gz"]].to_numpy(dtype=np.float32)

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    feats.extend([
        safe_corr(acc[:, 0], acc[:, 1]),
        safe_corr(acc[:, 0], acc[:, 2]),
        safe_corr(acc[:, 1], acc[:, 2]),
        safe_corr(gyr[:, 0], gyr[:, 1]),
        safe_corr(gyr[:, 0], gyr[:, 2]),
        safe_corr(gyr[:, 1], gyr[:, 2]),
    ])

    return feats


# =========================
# Windowing
# =========================
def assign_window_label(window_df: pd.DataFrame) -> str:
    # 用 majority vote 決定整個 window 的 action
    return str(window_df[LABEL_COL].mode().iloc[0])


def build_windows_from_df(df: pd.DataFrame, file_group: str):
    X, y, groups = [], [], []

    required_cols = set(SENSOR_COLS + [LABEL_COL])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少欄位: {missing}")

    # 僅保留有效標籤
    df = df.dropna(subset=SENSOR_COLS + [LABEL_COL]).reset_index(drop=True)
    df = df[df[LABEL_COL].isin(VALID_ACTIONS.keys())].reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        return X, y, groups

    # 如果資料裡本來就有 subject_id，可拿來做 group split
    if USE_GROUP_SPLIT and GROUP_COL in df.columns:
        unique_groups = df[GROUP_COL].dropna().unique().tolist()
        if len(unique_groups) == 1:
            group_value = str(unique_groups[0])
        else:
            # 同一檔內若有多個 subject_id，還是退回用 file_group
            group_value = file_group
    else:
        group_value = file_group

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_df = df.iloc[start:end]

        # 若 window 內標籤太混雜，可以選擇跳過
        label_counts = window_df[LABEL_COL].value_counts(normalize=True)
        if label_counts.iloc[0] < 0.8:
            continue

        feats = extract_features(window_df)
        label = assign_window_label(window_df)

        X.append(feats)
        y.append(label)
        groups.append(group_value)

    return X, y, groups


# =========================
# Loading data
# =========================
def load_dataset():
    csv_files = sorted(glob.glob(DATA_GLOB))
    if not csv_files:
        raise FileNotFoundError(f"找不到資料檔：{DATA_GLOB}")

    all_X, all_y, all_groups = [], [], []

    for csv_path in csv_files:
        print(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)

        file_group = os.path.splitext(os.path.basename(csv_path))[0]
        X_part, y_part, g_part = build_windows_from_df(df, file_group)

        all_X.extend(X_part)
        all_y.extend(y_part)
        all_groups.extend(g_part)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)
    groups = np.array(all_groups)

    return X, y, groups


# =========================
# Main
# =========================
def main():
    X, y_text, groups = load_dataset()

    if len(X) == 0:
        raise RuntimeError("沒有可用 windows。請檢查資料量、標籤純度與 WINDOW_SIZE。")

    print(f"Total windows: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print("Class distribution:")
    print(pd.Series(y_text).value_counts())

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    # 優先做 group split，避免同一人/同一檔案的片段同時進 train 和 test
    do_group_split = USE_GROUP_SPLIT and len(np.unique(groups)) >= 2

    if do_group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print("Using GroupShuffleSplit")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print("Using stratified random split")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 儲存
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    meta = {
        "sensor_cols": SENSOR_COLS,
        "label_col": LABEL_COL,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "valid_actions": VALID_ACTIONS,
        "feature_count": int(X.shape[1]),
        "group_split_used": bool(do_group_split),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved encoder -> {ENCODER_PATH}")
    print(f"Saved meta -> {META_PATH}")


if __name__ == "__main__":
    main()