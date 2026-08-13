from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def leave_one_video_out_appearance(examples: tuple[dict, ...]) -> dict:
    videos = sorted({str(item["video_name"]) for item in examples})
    folds = []
    for held_out in videos:
        train = [item for item in examples if item["video_name"] != held_out]
        test = [item for item in examples if item["video_name"] == held_out]
        if not train or not test or len({item["label"] for item in train}) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0),
        )
        model.fit(np.asarray([item["feature"] for item in train]), np.asarray([item["label"] for item in train]))
        truth = np.asarray([item["label"] for item in test])
        prediction = model.predict(np.asarray([item["feature"] for item in test]))
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, prediction, pos_label=1, average="binary", zero_division=0,
        )
        folds.append({
            "held_out_video": held_out,
            "train_examples": len(train),
            "test_examples": len(test),
            "test_positive": int(truth.sum()),
            "test_negative": int((truth == 0).sum()),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "feature_set": "shirt_hsv_histogram",
        "folds": folds,
        "mean_balanced_accuracy": float(np.mean([item["balanced_accuracy"] for item in folds])) if folds else None,
        "acceptance_rule": "do not integrate unless every held-out video materially beats 0.5 balanced accuracy",
    }
