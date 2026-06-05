from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


def softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    best = max(scores.values())
    exps = {label: math.exp(value - best) for label, value in scores.items()}
    total = sum(exps.values()) or 1.0
    return {label: value / total for label, value in exps.items()}


class ForestModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.classes = [str(item) for item in payload["classes"]]
        self.trees = list(payload["trees"])

    def _tree_proba(self, tree: dict[str, object], features: Sequence[float]) -> list[float]:
        children_left = tree["children_left"]
        children_right = tree["children_right"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        value = tree["value"]
        node = 0
        while children_left[node] != -1:
            idx = feature[node]
            if features[idx] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        probs = [float(v) for v in value[node]]
        total = sum(probs)
        if total <= 0.0:
            return [1.0 / len(self.classes)] * len(self.classes)
        return [v / total for v in probs]

    def predict_proba(self, features: Sequence[float]) -> dict[str, float]:
        sums = [0.0] * len(self.classes)
        for tree in self.trees:
            probs = self._tree_proba(tree, features)
            for idx, value in enumerate(probs):
                sums[idx] += value
        total_trees = max(1, len(self.trees))
        return {label: sums[idx] / total_trees for idx, label in enumerate(self.classes)}

    def predict(self, features: Sequence[float]) -> str:
        probs = self.predict_proba(features)
        return max(self.classes, key=lambda label: probs.get(label, 0.0))


class StandardScaler:
    def __init__(self, mean: Sequence[float], scale: Sequence[float]) -> None:
        self.mean = [float(value) for value in mean]
        self.scale = [float(value) if abs(float(value)) > 1e-12 else 1.0 for value in scale]

    def transform_one(self, features: Sequence[float]) -> list[float]:
        return [(float(value) - self.mean[idx]) / self.scale[idx] for idx, value in enumerate(features)]


class LuckFoxModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.labels = [str(item) for item in payload["labels"]]
        self.display_labels = {str(k): str(v) for k, v in payload.get("display_labels", {}).items()}
        self.config = dict(payload["config"])
        self.scaler = StandardScaler(payload["scaler"]["mean"], payload["scaler"]["scale"])
        self.active_model = ForestModel(payload["active_model"])
        self.action_model = ForestModel(payload["action_model"])

    @classmethod
    def load(cls, path: str | Path) -> "LuckFoxModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def transform(self, features: Sequence[float]) -> list[float]:
        return self.scaler.transform_one(features)

    def active_probability(self, features: Sequence[float]) -> float:
        probs = self.active_model.predict_proba(self.transform(features))
        return probs.get("Active", 0.0)

    def action_prediction(self, features: Sequence[float]) -> tuple[str, float]:
        probs = self.action_model.predict_proba(self.transform(features))
        label = max(probs, key=probs.get)
        return label, probs[label]
