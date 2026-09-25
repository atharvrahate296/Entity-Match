#!/usr/bin/env python3
"""
Train the entity-resolution matching model on the training split.

Usage (from code/EntityMatch/):
    python3 src/train.py --train-dir /path/to/dataset/train --model-out models/model.joblib

Pipeline:
  1. Load train_source1/2/3.tsv + train_ground_truth.tsv
  2. Blocking: generate candidate pairs (src/blocking.py)
  3. Label each candidate pair against ground truth (1 = true match, 0 = not)
  4. Group-split by Source-1 entity into train/val (no S1 leaks across the split)
  5. Feature engineering (src/features.py)
  6. Train HistGradientBoostingClassifier (src/model.py)
  7. Tune the decision threshold on the validation split to maximize macro F_0.5
  8. Persist {classifier, threshold} to --model-out

Also prints validation-set blocking recall (upper bound on achievable recall)
and the tuned macro F_0.5, so you can see where time is best spent next.
"""

import argparse
import os
import random

import joblib
import numpy as np
import pandas as pd

from blocking import generate_candidates
from data_io import build_lookup, load_ground_truth, load_source
from features import build_features
from model import train_classifier_batches, tune_threshold_from_probs


CANDIDATE_BATCH = 50_000


def group_split(s1_ids, val_frac=0.2, seed=42):
    ids = list(s1_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--model-out", default="code/EntityMatch/models/model.joblib")
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    s1 = load_source(os.path.join(args.train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(args.train_dir, "train_source3.tsv"))
    truth_map = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    all_s1_ids = s1["entity_id"].tolist()

    print(f"[train] {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3 records")

    print("[train] generating candidates (blocking)...")
    generate_candidates(s1, s2, s3, out_path=os.path.join(args.train_dir, "candidate_pairs.tsv"))
    candidate_path = os.path.join(args.train_dir, "candidate_pairs.tsv")

    # Blocking recall ceiling: fraction of true matches that made it into candidates.
    total_true = sum(len(v) for v in truth_map.values())
    hit_ids = set()
    for chunk in pd.read_csv(candidate_path, sep="\t", chunksize=CANDIDATE_BATCH):
        hit_ids.update(set(chunk["source1_entity_id"]) & set(truth_map))
    hit = len(hit_ids)
    recall_ceiling = hit / total_true if total_true else 1.0
    print(f"[train] blocking recall ceiling: {recall_ceiling:.4f} ({hit}/{total_true})")

    train_ids, val_ids = group_split(all_s1_ids, args.val_frac)

    s1_lookup = build_lookup(s1)
    other_lookup = build_lookup(s2)
    other_lookup.update(build_lookup(s3))

    def feature_batches(split_ids):
        for chunk in pd.read_csv(candidate_path, sep="\t", chunksize=CANDIDATE_BATCH):
            chunk = chunk[chunk["source1_entity_id"].isin(split_ids)].copy()
            if chunk.empty:
                continue
            chunk["label"] = [
                int(oid in truth_map.get(s1id, set()))
                for s1id, oid in zip(chunk["source1_entity_id"], chunk["candidate_entity_id"])
            ]
            yield build_features(chunk, s1_lookup, other_lookup), chunk["label"].to_numpy(dtype=np.int8)

    print("[train] streaming feature batches...")
    clf = train_classifier_batches(feature_batches(train_ids))

    val_rows = []
    val_probs = []
    val_count = 0
    for features, _ in feature_batches(val_ids):
        val_probs.extend(clf.predict_proba(features)[:, 1])
        val_count += len(features)
    for chunk in pd.read_csv(candidate_path, sep="\t", chunksize=CANDIDATE_BATCH):
        val_rows.append(chunk[chunk["source1_entity_id"].isin(val_ids)])
    val_pairs_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    val_truth = {k: v for k, v in truth_map.items() if k in val_ids}
    threshold, val_score = tune_threshold_from_probs(
        np.asarray(val_probs), val_pairs_df, val_truth, val_ids
    )
    print(f"[train] validation pairs: {val_count}")
    print(f"[train] tuned threshold={threshold:.2f} -> validation macro F_0.5={val_score:.4f}")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump({"classifier": clf, "threshold": threshold}, args.model_out)
    print(f"[train] model saved to {args.model_out}")


if __name__ == "__main__":
    main()
