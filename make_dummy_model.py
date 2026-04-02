import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

classes = [
    "db_bench_press",
    "one_arm_db_row",
    "db_shoulder_press",
    "db_biceps_curl",
    "db_triceps_extension",
    "db_squat",
    "db_rdl",
    "weighted_crunch",
]

# feature dimension
n_features = 70

X = np.random.randn(500, n_features)
y = np.random.choice(classes, 500)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier()
model.fit(X, y_encoded)

joblib.dump(model, "action_rf_model.pkl")
joblib.dump(encoder, "action_label_encoder.pkl")

meta = {
    "window_size": 100,
    "step_size": 50,
    "valid_actions": {
        "db_bench_press": "啞鈴臥推",
        "one_arm_db_row": "單手啞鈴划船",
        "db_shoulder_press": "啞鈴肩推",
        "db_biceps_curl": "啞鈴二頭彎舉",
        "db_triceps_extension": "啞鈴三頭彎舉",
        "db_squat": "啞鈴深蹲",
        "db_rdl": "啞鈴羅馬尼亞硬舉",
        "weighted_crunch": "啞鈴負重卷腹",
    }
}

with open("action_model_meta.json", "w") as f:
    json.dump(meta, f)

print("Dummy model created")