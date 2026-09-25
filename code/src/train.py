#!/usr/bin/env python3
"""Train the entity-resolution matcher — laptop-safe, batched, stratified.

Why sampling? The full train set is 2.2M S1 x 10M S2/S3 records; blocking
all of it yields ~60M candidate pairs, which no 16GB laptop can featurize
in memory. The classifier learns a *pairwise* match function, so a
country-stratified sample of S1 entities (default 100k train / 20k val,
~3-4M pairs) trains the same function with bounded RAM. Distribution is
preserved by splitting and sampling *within each country*.

Flow (each stage streams; peak RAM stays ~6-8GB on 16GB machines):
  1. Load S1 ids + country, stratified group-split -> train/val S1 ids.
  2. Stratified sample caps -> load truth only for sampled ids.
  3. Load S2/S3 frames once for blocking refs; slice sampled S1 rows.
  4. Blocking on the sample -> scored per-pair temp file.
  5. Cache only needed entities (sampled S1 + referenced S2/S3), free frames.
  6. Stream pairs -> features (threaded) -> in-batch-shuffled SGD partial_fit.
  7. Single streaming validation pass -> vectorized macro-F0.5 threshold tune.
  8. Persist {classifier, threshold} bundle.

Run from the solution root:  python code/src/train.py
Smoke test (minutes):       python code/src/train.py --s1-limit 2000
"""

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

# Allow running as `python code/src/train.py` from the solution root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from blocking import generate_candidates
from data_io import (build_entity_cache, collect_candidate_oids, load_ground_truth,
                     load_source, write_candidate_id_list)
from features import build_features
from model import train_classifier_batches, tune_threshold_from_probs

CANDIDATE_BATCH = 50_000


def stratified_split(s1_ids, countries, val_frac=0.2, seed=42):
    """Group-wise split by S1 entity, stratified by country (no S1 leakage)."""
    rng = random.Random(seed)
    train_ids, val_ids = set(), set()
    by_country = {}
    for s1, c in zip(s1_ids, countries):
        by_country.setdefault(c, []).append(s1)
    for c, ids in by_country.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_frac))
        val_ids.update(ids[:n_val])
        train_ids.update(ids[n_val:])
    return train_ids, val_ids


def stratified_sample(ids, id2country, n_max, seed):
    """Cap an id set to n_max, preserving country proportions."""
    ids = list(ids)
    if len(ids) <= n_max:
        return set(ids)
    rng = random.Random(seed)
    by_country = {}
    for s1 in ids:
        by_country.setdefault(id2country[s1], []).append(s1)
    out = set()
    total = len(ids)
    for c, grp in by_country.items():
        k = max(1, round(len(grp) / total * n_max))
        rng.shuffle(grp)
        out.update(grp[:k])
    # Fix rounding drift.
    ids_rest = [s for s in ids if s not in out]
    rng.shuffle(ids_rest)
    while len(out) < n_max and ids_rest:
        out.add(ids_rest.pop())
    while len(out) > n_max:
        out.pop()
    return out


def log_dist(tag, ids, id2country):
    from collections import Counter
    print(f"[train] {tag}: {len(ids)} S1 {dict(Counter(id2country[s] for s in ids))}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--model-out", default="code/models/model.joblib")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-s1", type=int, default=100_000,
                    help="Stratified cap on training S1 entities (bounds RAM/time).")
    ap.add_argument("--max-val-s1", type=int, default=20_000,
                    help="Stratified cap on validation S1 entities.")
    ap.add_argument("--s1-limit", type=int, default=0,
                    help="Smoke-test: use only first N S1 rows (0 = off).")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--feature-batch", type=int, default=CANDIDATE_BATCH)
    ap.add_argument("--workers", type=int, default=4,
                    help="Feature threads (rapidfuzz releases the GIL).")
    args = ap.parse_args()
    t_all = time.time()

    s1 = load_source(os.path.join(args.train_dir, "train_source1.tsv"))
    if args.s1_limit:
        s1 = s1.iloc[:args.s1_limit].copy()
        print(f"[train] SMOKE MODE: first {len(s1)} S1 rows", flush=True)
    all_s1_ids = s1["entity_id"].tolist()
    id2country = dict(zip(all_s1_ids, s1["country"].astype(str).tolist()))
    print(f"[train] S1 records: {len(s1)}", flush=True)

    train_ids, val_ids = stratified_split(
        all_s1_ids, [id2country[s] for s in all_s1_ids], args.val_frac, args.seed)
    train_ids = stratified_sample(train_ids, id2country, args.max_train_s1, args.seed)
    val_ids = stratified_sample(val_ids, id2country, args.max_val_s1, args.seed + 1)
    log_dist("train split", train_ids, id2country)
    log_dist("val split  ", val_ids, id2country)

    sample_ids = train_ids | val_ids
    truth_map = load_ground_truth(
        os.path.join(args.train_dir, "train_ground_truth.tsv"), keep_ids=sample_ids)
    print(f"[train] truth entries for sample: {len(truth_map)}", flush=True)

    print("[train] loading S2/S3 reference tables...", flush=True)
    s2 = load_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(args.train_dir, "train_source3.tsv"))
    print(f"[train] S2={len(s2)} S3={len(s3)}", flush=True)

    s1_sample = s1[s1["entity_id"].isin(sample_ids)].copy()
    del s1
    gc.collect()
    print(f"[train] blocking on {len(s1_sample)} sampled S1...", flush=True)
    scored_path = os.path.join(args.output_dir, "_train_candidates_scored.tsv")
    n_pairs, n_s1 = generate_candidates(
        s1_sample, s2, s3, out_path=scored_path, top_k=args.top_k)
    if n_pairs == 0:
        raise ValueError("blocking produced no candidates")

    # Blocking recall ceiling on the sample: fraction of ground-truth
    # (s1, match) pairs present in the candidate set (upper bound on recall).
    hit_pairs = 0
    total_true_pairs = sum(len(v) for v in truth_map.values())
    for chunk in pd.read_csv(scored_path, sep="\t", chunksize=args.feature_batch,
                             usecols=["source1_entity_id", "candidate_entity_id"]):
        hit_pairs += sum(
            1 for s1id, oid in zip(chunk["source1_entity_id"], chunk["candidate_entity_id"])
            if oid in truth_map.get(s1id, ()))
    print(f"[train] blocking recall ceiling: "
          f"{hit_pairs / max(total_true_pairs, 1):.4f} "
          f"({hit_pairs}/{total_true_pairs} true pairs covered)", flush=True)

    # Validator-format candidate list for the sampled S1 (debug artifact).
    write_candidate_id_list(
        scored_path, os.path.join(args.output_dir, "candidate_pairs_train.tsv"),
        list(sample_ids))

    # Cache only needed entities, then free the big frames.
    print("[train] building entity caches...", flush=True)
    oids = collect_candidate_oids(scored_path)
    print(f"[train] unique candidate S2/S3 ids: {len(oids)}", flush=True)
    s1_cache = build_entity_cache(s1_sample, sample_ids)
    del s1_sample
    gc.collect()
    ref_cache = build_entity_cache(s2, oids)
    del s2
    gc.collect()
    ref_cache.update(build_entity_cache(s3, oids))
    del s3
    gc.collect()
    print(f"[train] caches: S1={len(s1_cache)} refs={len(ref_cache)} "
          f"({time.time() - t_all:.0f}s elapsed)", flush=True)

    def feature_batches(split_ids):
        for chunk in pd.read_csv(scored_path, sep="\t", chunksize=args.feature_batch):
            chunk = chunk[chunk["source1_entity_id"].isin(split_ids)]
            if chunk.empty:
                continue
            s1a = chunk["source1_entity_id"].tolist()
            cands = chunk["candidate_entity_id"].tolist()
            y = np.array([int(oid in truth_map.get(s1id, ())) for s1id, oid in
                          zip(s1a, cands)], dtype=np.int8)
            X = build_features(chunk, s1_cache, ref_cache, workers=args.workers)
            yield X, y

    print("[train] streaming feature batches...", flush=True)
    # Cheap label-only pass for exact class balance (partial_fit rejects
    # class_weight="balanced", so pass explicit weights instead).
    n_tr_tot = n_tr_pos = 0
    for chunk in pd.read_csv(scored_path, sep="\t", chunksize=args.feature_batch,
                             usecols=["source1_entity_id", "candidate_entity_id"]):
        m = chunk["source1_entity_id"].isin(train_ids)
        if not m.any():
            continue
        sub = chunk[m]
        n_tr_tot += len(sub)
        n_tr_pos += sum(1 for s1id, oid in
                        zip(sub["source1_entity_id"], sub["candidate_entity_id"])
                        if oid in truth_map.get(s1id, ()))
    if n_tr_pos == 0:
        raise ValueError("no positive training pairs in sample; "
                         "raise --max-train-s1 or lower blocking thresholds")
    n_tr_neg = n_tr_tot - n_tr_pos
    cw = {0: n_tr_tot / (2 * n_tr_neg), 1: n_tr_tot / (2 * n_tr_pos)}
    print(f"[train] train pairs={n_tr_tot} pos={n_tr_pos} "
          f"({100.0 * n_tr_pos / n_tr_tot:.2f}%) class_weight={cw}", flush=True)
    clf, n_tr, n_pos = train_classifier_batches(feature_batches(train_ids),
                                                class_weight=cw)

    # Single streaming validation pass (no second full concat).
    print("[train] scoring validation pairs...", flush=True)
    v_s1, v_cand, v_prob = [], [], []
    for chunk in pd.read_csv(scored_path, sep="\t", chunksize=args.feature_batch):
        chunk = chunk[chunk["source1_entity_id"].isin(val_ids)]
        if chunk.empty:
            continue
        X = build_features(chunk, s1_cache, ref_cache, workers=args.workers)
        p = clf.predict_proba(X.to_numpy(dtype=np.float32))[:, 1]
        v_s1.extend(chunk["source1_entity_id"].tolist())
        v_cand.extend(chunk["candidate_entity_id"].tolist())
        v_prob.extend(p.tolist())
    print(f"[train] validation pairs: {len(v_prob)}", flush=True)
    val_truth = {k: v for k, v in truth_map.items() if k in val_ids}
    threshold, val_score = tune_threshold_from_probs(
        v_s1, v_cand, np.asarray(v_prob), val_truth, val_ids)
    print(f"[train] tuned threshold={threshold:.2f} -> validation macro F_0.5={val_score:.4f}",
          flush=True)

    del s1_cache, ref_cache
    gc.collect()
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump({"classifier": clf, "threshold": threshold,
                 "top_k": args.top_k, "seed": args.seed}, args.model_out)
    print(f"[train] model saved to {args.model_out} "
          f"(total {time.time() - t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
