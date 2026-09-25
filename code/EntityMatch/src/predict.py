#!/usr/bin/env python3
"""
Run inference on the test set and write the two required output files.

Usage (from code/EntityMatch/):
    python3 src/predict.py --test-dir /path/to/dataset/test \
        --model models/model.joblib --output-dir output

Writes:
    output/candidate_pairs.tsv   (blocking stage output -- what the model scores)
    output/matching_results.tsv  (final matches after thresholding -- leaderboard file)

`candidate_pairs.tsv` is written from exactly the same candidate set the
classifier scores in this run, and `matching_results.tsv` is filtered from
that set only, so matches are always a subset of candidates by construction.
"""

import argparse
import os

import joblib

from blocking import generate_candidates
from data_io import build_lookup, candidates_to_pairs_df, load_source, write_id_list_tsv
from features import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model", default="code/EntityMatch/models/model.joblib")
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    s1 = load_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(args.test_dir, "test_source3.tsv"))
    all_s1_ids = s1["entity_id"].tolist()

    print(f"[predict] {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3 test records")

    print("[predict] generating candidates (blocking)...")
    candidates = generate_candidates(s1, s2, s3)
    pairs_df = candidates_to_pairs_df(candidates)
    print(f"[predict] {len(pairs_df)} candidate pairs generated")

    candidate_map = {}
    for s1id, oid in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]):
        candidate_map.setdefault(s1id, set()).add(oid)

    write_id_list_tsv(
        os.path.join(args.output_dir, "candidate_pairs.tsv"),
        ["source1_entity_id", "candidate_entity_ids"],
        candidate_map,
        all_s1_ids,
    )
    print("[predict] wrote candidate_pairs.tsv")

    if len(pairs_df) == 0:
        # No candidates at all -> every S1 entity is a predicted singleton.
        write_id_list_tsv(
            os.path.join(args.output_dir, "matching_results.tsv"),
            ["source1_entity_id", "matched_entity_ids"],
            {},
            all_s1_ids,
        )
        print("[predict] wrote matching_results.tsv (no candidates found)")
        return

    bundle = joblib.load(args.model)
    clf, threshold = bundle["classifier"], bundle["threshold"]

    s1_lookup = build_lookup(s1)
    other_lookup = build_lookup(s2)
    other_lookup.update(build_lookup(s3))

    print("[predict] building features...")
    X = build_features(pairs_df, s1_lookup, other_lookup)
    probs = clf.predict_proba(X)[:, 1]

    match_map = {}
    for s1id, oid, p in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"], probs):
        if p >= threshold:
            match_map.setdefault(s1id, set()).add(oid)

    write_id_list_tsv(
        os.path.join(args.output_dir, "matching_results.tsv"),
        ["source1_entity_id", "matched_entity_ids"],
        match_map,
        all_s1_ids,
    )
    n_matched = sum(1 for v in match_map.values() if v)
    print(f"[predict] wrote matching_results.tsv ({n_matched}/{len(all_s1_ids)} S1 entities matched)")


if __name__ == "__main__":
    main()
