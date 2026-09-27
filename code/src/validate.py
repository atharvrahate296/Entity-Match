"""Reproduce train.py's exact held-out validation split, run real inference
on those Source-1 entities against the TRAIN reference tables (train_source2/
3.tsv -- that's where their true matches actually live), and write a
matching_results.tsv-format file you can score with score_predictions.py.

WHY THIS SCRIPT EXISTS: matching_results.tsv (from predict.py) is your
TEST-set predictions; train_ground_truth.tsv only has labels for TRAIN-set
entities. Their id spaces are unrelated -- scoring one against the other
(as in `score_predictions.py --predictions output/matching_results.tsv
--ground-truth dataset/train/train_ground_truth.tsv`) compares entities
that don't correspond to each other at all and produces a meaningless
number. This script generates a prediction file for entities that DO have
real ground truth: a slice of train data the model has genuinely never
been trained on (same split, same seed as train.py, so there's no leakage).

Usage:
    python code/src/validate.py
    python code/src/score_predictions.py \\
        --predictions output/validation/matching_results.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --only-ids output/validation/val_ids.txt

If you trained with non-default --val-frac / --seed / --max-val-s1, pass
the same values here so this reproduces the actual split train.py used.

⚠️ THIS IS EASY TO GET WRONG SILENTLY: if these values don't match what
train.py was actually run with, this script still runs, still writes a
file, and score_predictions.py still prints numbers -- they'll just be
scoring the WRONG 40,000 (or whatever N) entities against the model,
which is worse than an obvious crash because nothing LOOKS wrong. Always
double check the --max-val-s1 (and --val-frac/--seed if changed) here
match your most recent train.py invocation before trusting the score.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from blocking import generate_candidates
from data_io import (build_entity_cache, build_entity_cache_from_path,
                     load_source, write_id_list_tsv)
from features import build_features
from train import stratified_split, stratified_sample

CHUNK = 50_000
CACHE_CHUNK = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--model", default="code/models/model.joblib")
    ap.add_argument("--output-dir", default="output/validation")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Must match the value train.py was run with.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Must match the value train.py was run with.")
    ap.add_argument("--max-val-s1", type=int, default=40_000,
                    help="MUST MATCH the value train.py was actually run with "
                         "(train.py currently defaults to 40_000). A mismatch "
                         "here silently reproduces the WRONG validation split "
                         "-- score_predictions.py will still run and print "
                         "numbers, but they won't reflect your actual model.")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Must match blocking.py's module-level TOP_K and "
                         "train.py's/predict.py's --top-k.")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t_all = time.time()

    s1 = load_source(os.path.join(args.train_dir, "train_source1.tsv"))
    all_s1_ids = s1["entity_id"].tolist()
    id2country = dict(zip(all_s1_ids, s1["country"].astype(str).tolist()))

    # Same split function, same seed, same args as train.py -> these are
    # exactly the S1 entities the saved model never trained on.
    _, val_ids = stratified_split(
        all_s1_ids, [id2country[s] for s in all_s1_ids], args.val_frac, args.seed)
    val_ids = stratified_sample(val_ids, id2country, args.max_val_s1, args.seed + 1)
    print(f"[validate] reproduced {len(val_ids)} held-out validation S1 ids", flush=True)

    s1_val = s1[s1["entity_id"].isin(val_ids)].copy()
    val_ids_list = s1_val["entity_id"].tolist()
    s1_ids_by_country = {}
    for sid, c in zip(val_ids_list, s1_val["country"].astype(str).tolist()):
        s1_ids_by_country.setdefault(c, set()).add(str(sid))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "val_ids.txt"), "w", encoding="utf-8") as f:
        for vid in val_ids_list:
            f.write(f"{vid}\n")

    s2_path = os.path.join(args.train_dir, "train_source2.tsv")
    s3_path = os.path.join(args.train_dir, "train_source3.tsv")

    print("[validate] generating candidates (blocking) for validation S1...", flush=True)
    scored_path = os.path.join(args.output_dir, "_val_candidates_scored.tsv")
    n_pairs, n_s1 = generate_candidates(
        s1_val, s2_path, s3_path, out_path=scored_path, top_k=args.top_k)
    print(f"[validate] {n_pairs} candidate pairs for {n_s1}/{len(val_ids_list)} S1",
          flush=True)

    bundle = joblib.load(args.model)
    clf, threshold = bundle["classifier"], bundle["threshold"]
    print(f"[validate] model threshold={threshold:.2f}", flush=True)

    s1_cache = build_entity_cache(s1_val, val_ids_list)
    del s1, s1_val
    gc.collect()

    match_map = {}
    for country, s1_ids_this in s1_ids_by_country.items():
        c_t = time.time()
        oids_this = set()
        for chunk in pd.read_csv(scored_path, sep="\t", chunksize=CHUNK,
                                 usecols=["source1_entity_id", "candidate_entity_id"]):
            m = chunk["source1_entity_id"].isin(s1_ids_this)
            if m.any():
                oids_this.update(chunk.loc[m, "candidate_entity_id"].tolist())

        ref_cache = build_entity_cache_from_path(s2_path, oids_this, chunksize=CACHE_CHUNK)
        gc.collect()
        ref_cache.update(build_entity_cache_from_path(s3_path, oids_this, chunksize=CACHE_CHUNK))
        gc.collect()
        del oids_this

        for chunk in pd.read_csv(scored_path, sep="\t", chunksize=CHUNK):
            chunk = chunk[chunk["source1_entity_id"].isin(s1_ids_this)]
            if chunk.empty:
                continue
            X = build_features(chunk, s1_cache, ref_cache, workers=args.workers)
            probs = clf.predict_proba(X.to_numpy(dtype=np.float32))[:, 1]
            for s1id, oid, p in zip(chunk["source1_entity_id"],
                                    chunk["candidate_entity_id"], probs):
                if p >= threshold:
                    match_map.setdefault(s1id, set()).add(oid)
        del ref_cache
        gc.collect()
        print(f"[validate] {country}: done ({time.time() - c_t:.0f}s)", flush=True)

    out_path = os.path.join(args.output_dir, "matching_results.tsv")
    write_id_list_tsv(
        out_path,
        ["source1_entity_id", "matched_entity_ids"],
        match_map,
        val_ids_list,
    )
    print(f"[validate] wrote {out_path} ({time.time() - t_all:.0f}s total)", flush=True)
    try:
        os.remove(scored_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()