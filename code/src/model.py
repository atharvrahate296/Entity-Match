import time

import numpy as np
from sklearn.linear_model import SGDClassifier

from features import FEATURE_COLUMNS


def train_classifier(X_train, y_train):
    clf = _new_classifier()
    clf.fit(X_train[FEATURE_COLUMNS], y_train)
    return clf


def _new_classifier(class_weight=None):
    # NOTE: partial_fit does not accept class_weight="balanced"; the caller
    # estimates the pos/neg ratio in a cheap label-only pass and passes an
    # explicit {0: w0, 1: w1} dict (fully supported by partial_fit).
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight=class_weight,
        average=True,
        learning_rate="optimal",
        eta0=0.0,
        tol=None,
        random_state=42,
    )


def train_classifier_batches(batches, class_weight=None, log_every=10):
    """Single streaming pass over (X, y) batches with in-batch shuffling.

    Batches arrive in candidate-file order (country-grouped, highly
    correlated); shuffling inside each batch keeps SGD updates stable.
    Returns (clf, n_pairs, n_pos).
    """
    clf = _new_classifier(class_weight)
    first_batch = True
    n_pairs = n_pos = 0
    t0 = time.time()
    for bi, (X_batch, y_batch) in enumerate(batches):
        if len(X_batch) == 0:
            continue
        X = X_batch[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = np.asarray(y_batch, dtype=np.int8)
        rng = np.random.default_rng(42 + bi)
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]
        clf.partial_fit(X, y, classes=np.array([0, 1]) if first_batch else None)
        first_batch = False
        n_pairs += len(y)
        n_pos += int(y.sum())
        if (bi + 1) % log_every == 0:
            print(f"[train]   batch {bi + 1}: {n_pairs} pairs "
                  f"({n_pos} pos, {time.time() - t0:.0f}s)", flush=True)
    if first_batch:
        raise ValueError("no training candidates were generated")
    print(f"[train] classifier done: {n_pairs} pairs, {n_pos} positives "
          f"({100.0 * n_pos / max(n_pairs, 1):.2f}% pos, {time.time() - t0:.0f}s)",
          flush=True)
    return clf, n_pairs, n_pos


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f05(pred_map, truth_map, all_s1_ids):
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


def tune_threshold_from_probs(s1_ids, cand_ids, probs, truth_map, val_ids, thresholds=None):
    """Vectorized macro-F0.5 threshold search (exact, no per-threshold dicts).

    s1_ids / cand_ids / probs are parallel arrays over validation pairs.
    Per-threshold work is numpy bincount over factorized entity codes, so
    46 thresholds over ~1M pairs take seconds instead of tens of minutes.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.02)
    s1_ids = np.asarray(s1_ids, dtype=object)
    cand_ids = np.asarray(cand_ids, dtype=object)
    probs = np.asarray(probs, dtype=np.float64)
    val_ids = list(val_ids)
    codes, uniq = pd_factorize(s1_ids)
    n_ent = len(uniq)
    truth_count = np.zeros(n_ent, dtype=np.int64)
    code_of = {s: i for i, s in enumerate(uniq)}
    for s1, truth in truth_map.items():
        i = code_of.get(s1)
        if i is not None:
            truth_count[i] = len(truth)
    # Per-pair true-positive flag, computed once.
    tp_flag = np.zeros(len(s1_ids), dtype=np.int64)
    for i, (s1, oid) in enumerate(zip(s1_ids, cand_ids)):
        t = truth_map.get(s1)
        if t is not None and oid in t:
            tp_flag[i] = 1
    # Entities with no ground truth and no prediction score 1.0: credit them
    # at every threshold via the zero-prediction baseline.
    empty_truth = (truth_count == 0)

    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        keep = probs >= t
        if keep.any():
            kc = codes[keep]
            pred_count = np.bincount(kc, minlength=n_ent).astype(np.float64)
            tp_count = np.bincount(kc, weights=tp_flag[keep], minlength=n_ent)
        else:
            pred_count = np.zeros(n_ent)
            tp_count = np.zeros(n_ent)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.divide(tp_count, pred_count, out=np.zeros_like(tp_count, dtype=float),
                             where=pred_count > 0)
            rec = np.divide(tp_count, truth_count, out=np.zeros_like(tp_count, dtype=float),
                            where=truth_count > 0)
        b2 = 0.25
        denom = b2 * prec + rec
        f = np.divide((1 + b2) * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
        # No-prediction entities: 1.0 iff also no truth, else 0.0 (already 0).
        nopred = pred_count == 0
        f[nopred & empty_truth] = 1.0
        # Restrict the mean to validation entities (uniq is subset of val).
        # Entities in val_ids with no pairs at all score via the same rule.
        score = float(f.mean()) if n_ent else 0.0
        # Account for val entities absent from pairs: they have no pred;
        # 1.0 if empty truth else 0.0.
        n_missing = len(val_ids) - n_ent
        if n_missing > 0:
            n_empty_missing = sum(1 for s in val_ids
                                  if s not in code_of and not truth_map.get(s))
            score = (score * n_ent + n_empty_missing) / len(val_ids)
        if score > best_score:
            best_score, best_t = score, float(t)
    return float(best_t), float(best_score)


def pd_factorize(arr):
    uniq, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int64), uniq
