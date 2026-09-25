"""Incremental matching model and F_0.5 threshold selection."""

import numpy as np
from sklearn.linear_model import SGDClassifier

from features import FEATURE_COLUMNS


def train_classifier(X_train, y_train):
    clf = _new_classifier()
    clf.fit(X_train[FEATURE_COLUMNS], y_train)
    return clf


def _new_classifier():
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=1,
        tol=None,
        average=True,
        random_state=42,
    )


def train_classifier_batches(batches):
    """Fit incrementally so all candidate features never coexist in RAM."""
    clf = _new_classifier()
    first_batch = True
    for X_batch, y_batch in batches:
        if len(X_batch) == 0:
            continue
        clf.partial_fit(
            X_batch[FEATURE_COLUMNS].astype(np.float32, copy=False),
            y_batch,
            classes=np.array([0, 1]) if first_batch else None,
        )
        first_batch = False
    if first_batch:
        raise ValueError("no training candidates were generated")
    return clf


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f05(pred_map, truth_map, all_s1_ids):
    """
    pred_map / truth_map: source1_entity_id -> set(matched ids)
    all_s1_ids: the full set of Source-1 ids the average is taken over
    (singletons with an empty truth set score 1.0 when predicted empty).
    """
    scores = []
    for s1 in all_s1_ids:
        pred = pred_map.get(s1, set())
        truth = truth_map.get(s1, set())
        if not truth and not pred:
            scores.append(1.0)
            continue
        if not pred:
            scores.append(0.0)
            continue
        tp = len(pred & truth)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(truth) if truth else 0.0
        scores.append(f_beta(precision, recall, beta=0.5))
    return float(np.mean(scores)) if scores else 0.0


def tune_threshold(clf, X_val, val_pairs_df, truth_map, all_s1_ids, thresholds=None):
    """Search a probability threshold that maximizes macro F_0.5 on validation."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.02)
    probs = clf.predict_proba(X_val[FEATURE_COLUMNS].astype(np.float32, copy=False))[:, 1]

    return tune_threshold_from_probs(probs, val_pairs_df, truth_map, all_s1_ids, thresholds)


def tune_threshold_from_probs(probs, val_pairs_df, truth_map, all_s1_ids, thresholds=None):
    """Tune from cached probabilities, avoiding repeated model inference."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.02)

    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred_map = {}
        keep = probs >= t
        for s1, other, k in zip(
            val_pairs_df["source1_entity_id"][keep],
            val_pairs_df["candidate_entity_id"][keep],
            keep[keep],
        ):
            pred_map.setdefault(s1, set()).add(other)
        score = macro_f05(pred_map, truth_map, all_s1_ids)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t), float(best_score)
