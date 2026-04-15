# train_action_model.py
# 需求:
# pip install numpy pandas scikit-learn joblib

import os
import glob
import json
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# =========================
# 可調參數
# =========================

# 你的六軸欄位
SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

# 若開啟，使用 subject 做 group split（避免同一人同時出現在 train/test）
USE_GROUP_SPLIT = True
TRAIN_SUBJECT_IDS = {"thomas", "thomas_2"}
TEST_SUBJECT_IDS = {"kevin"}

# directory-based dataset layout:
# data/<subject>/db_*/set*/rep*.csv
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SET_DIR_RE = re.compile(r"^set\d+$", re.IGNORECASE)
SET_W_DIR_RE = re.compile(r"^set\d+_w$", re.IGNORECASE)
REP_FILE_RE = re.compile(r"^rep\d+.*\.csv$", re.IGNORECASE)
REST_FILE_RE = re.compile(r"^rest.*\.csv$", re.IGNORECASE)

# 取樣率假設 50 Hz；2 秒 window = 100 samples
WINDOW_SIZE = 100
STEP_SIZE = 50

RANDOM_STATE = 42

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "action_rf_model.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "action_label_encoder.pkl")
META_PATH = os.path.join(PROJECT_ROOT, "models", "action_model_meta.json")
PORTABLE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "action_rf_model_portable.json")

# 八類標籤映射，方便你統一資料名稱
VALID_ACTIONS = {
    "no_action": "無動作",
    "db_bench_press": "啞鈴臥推",
    "one_arm_db_row": "單手啞鈴划船",
    "db_shoulder_press": "啞鈴肩推",
    "db_biceps_curl": "啞鈴二頭彎舉",
    "db_triceps_extension": "啞鈴三頭彎舉",
    "db_squat": "啞鈴深蹲",
    "db_rdl": "啞鈴羅馬尼亞硬舉",
    "db_weighted_crunch": "啞鈴負重卷腹",
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
def build_windows_from_df(df: pd.DataFrame, label: str, group_value: str):
    X, y, groups = [], [], []

    required_cols = set(SENSOR_COLS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少欄位: {missing}")

    # only keep valid sensor rows
    df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        return X, y, groups

    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_df = df.iloc[start:end]

        feats = extract_features(window_df)

        X.append(feats)
        y.append(label)
        groups.append(group_value)

    return X, y, groups


# =========================
# Loading data
# =========================
def discover_training_csv_files() -> list[tuple[str, str, str, str]]:
    selected = []
    skipped_w_sets = 0
    skipped_other = 0
    subject_dirs = []

    all_subject_ids = sorted(TRAIN_SUBJECT_IDS | TEST_SUBJECT_IDS)
    for subject_id in all_subject_ids:
        subject_dir = os.path.join(DATA_ROOT, subject_id)
        if os.path.isdir(subject_dir):
            subject_dirs.append(subject_dir)

    if not subject_dirs:
        raise FileNotFoundError(
            f"在 {DATA_ROOT} 找不到受試者資料夾：{all_subject_ids}"
        )

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        for action_name in sorted(os.listdir(subject_dir)):
            action_dir = os.path.join(subject_dir, action_name)
            if not os.path.isdir(action_dir):
                continue

            # long-rest folder under subject root -> no_action
            if action_name == "big_rest":
                for root, _, files in os.walk(action_dir):
                    set_name = os.path.basename(root)
                    for name in sorted(files):
                        if not REST_FILE_RE.match(name):
                            continue
                        selected.append((os.path.join(root, name), subject_id, "no_action", set_name))
                continue

            if not action_name.startswith("db_") and action_name != "one_arm_db_row":
                continue

            if action_name not in VALID_ACTIONS:
                skipped_other += 1
                continue

            for set_name in sorted(os.listdir(action_dir)):
                set_dir = os.path.join(action_dir, set_name)
                if not os.path.isdir(set_dir):
                    continue
                if set_name.startswith("rest_after_set"):
                    for rest_name in sorted(os.listdir(set_dir)):
                        rest_path = os.path.join(set_dir, rest_name)
                        if not os.path.isfile(rest_path):
                            continue
                        if not REST_FILE_RE.match(rest_name):
                            continue
                        selected.append((rest_path, subject_id, "no_action", set_name))
                    continue
                if SET_W_DIR_RE.match(set_name):
                    skipped_w_sets += 1
                    continue
                if not SET_DIR_RE.match(set_name):
                    continue

                for rep_name in sorted(os.listdir(set_dir)):
                    rep_path = os.path.join(set_dir, rep_name)
                    if not os.path.isfile(rep_path):
                        continue
                    if not REP_FILE_RE.match(rep_name):
                        continue
                    selected.append((rep_path, subject_id, action_name, set_name))

    if not selected:
        raise FileNotFoundError(
            "找不到符合規則的訓練資料（data/<subject>/db_*/set*/rep*.csv，且排除 set*_w）"
        )

    print(f"[DATA] subjects: {[os.path.basename(x) for x in subject_dirs]}")
    print(f"[DATA] selected rep csv: {len(selected)}")
    print(f"[DATA] skipped set*_w dirs: {skipped_w_sets}")
    print(f"[DATA] skipped unknown action dirs: {skipped_other}")
    return selected


def load_dataset():
    csv_items = discover_training_csv_files()

    all_X, all_y, all_groups, all_subjects = [], [], [], []

    for csv_path, subject_id, action_name, set_name in csv_items:
        print(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
        group_value = subject_id if USE_GROUP_SPLIT else f"{subject_id}_{set_name}"
        X_part, y_part, g_part = build_windows_from_df(df, action_name, group_value)

        all_X.extend(X_part)
        all_y.extend(y_part)
        all_groups.extend(g_part)
        all_subjects.extend([subject_id] * len(X_part))

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)
    groups = np.array(all_groups)
    subjects = np.array(all_subjects)

    return X, y, groups, subjects


def export_portable_rf_json(model: RandomForestClassifier, classes: list[str], out_path: str) -> None:
    payload = {
        "model_type": "portable_rf_v1",
        "classes": [str(c) for c in classes],
        "n_features": int(model.n_features_in_),
        "trees": [],
    }

    for estimator in model.estimators_:
        tree = estimator.tree_
        nodes = []
        for i in range(tree.node_count):
            nodes.append(
                {
                    "left": int(tree.children_left[i]),
                    "right": int(tree.children_right[i]),
                    "feature": int(tree.feature[i]),
                    "threshold": float(tree.threshold[i]),
                    "value": [float(x) for x in tree.value[i][0]],
                }
            )
        payload["trees"].append(nodes)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


# =========================
# Main
# =========================
def main():
    X, y_text, groups, subjects = load_dataset()

    if len(X) == 0:
        raise RuntimeError("沒有可用 windows。請檢查資料量、標籤純度與 WINDOW_SIZE。")

    print(f"Total windows: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print("Class distribution:")
    print(pd.Series(y_text).value_counts())

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    # 固定 subject split：thomas/thomas_2 -> train, kevin -> test
    train_mask = np.isin(subjects, list(TRAIN_SUBJECT_IDS))
    test_mask = np.isin(subjects, list(TEST_SUBJECT_IDS))
    do_group_split = False

    if np.any(train_mask) and np.any(test_mask):
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        print(f"Using fixed subject split train={sorted(TRAIN_SUBJECT_IDS)} test={sorted(TEST_SUBJECT_IDS)}")
    else:
        missing = []
        if not np.any(train_mask):
            missing.append("train subjects")
        if not np.any(test_mask):
            missing.append("test subjects")
        raise RuntimeError(
            f"固定 subject split 無法建立，缺少: {', '.join(missing)}. "
            f"train={sorted(TRAIN_SUBJECT_IDS)} test={sorted(TEST_SUBJECT_IDS)}"
        )

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
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    export_portable_rf_json(model, classes=list(le.classes_), out_path=PORTABLE_MODEL_PATH)

    meta = {
        "sensor_cols": SENSOR_COLS,
        "label_source": "folder_name(db_*)",
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "valid_actions": VALID_ACTIONS,
        "feature_count": int(X.shape[1]),
        "group_split_used": bool(do_group_split),
        "train_subject_ids": sorted(TRAIN_SUBJECT_IDS),
        "test_subject_ids": sorted(TEST_SUBJECT_IDS),
        "portable_model_path": PORTABLE_MODEL_PATH,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved encoder -> {ENCODER_PATH}")
    print(f"Saved portable model -> {PORTABLE_MODEL_PATH}")
    print(f"Saved meta -> {META_PATH}")


if __name__ == "__main__":
    main()
