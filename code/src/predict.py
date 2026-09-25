#!/usr/bin/env python3
"""Generate test-set predictions — streams so a 16GB laptop survives.

Blocking runs over the full test S1 (no sampling here: every S1 needs an
output row), then pairs are scored in chunks and only *positive* matches
are accumulated. The full candidate map is never materialized (the old
code built a dict of every pair just to count them).

Outputs (validator format, one row per S1):
  output/candidate_pairs.tsv  (source1_entity_id, candidate_entity_ids)
  output/matching_results.tsv (source1_entity_id, matched_entity_ids)

Run from the solution root:  python code/src/predict.py
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# Allow running as `python code/src/predict.py` from the solution root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from blocking import generate_candidates
from data_io import (build_entity_cache, collect_candidate_oids, load_source,
                     write_candidate_id_list, write_id_list_tsv)
from features import build_features

CHUNK = 50_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model", default="code/models/model.joblib")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Must be >= the training --top-k for consistent recall.")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t_all = time.time()

    s1 = load_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(args.test_dir, "test_source3.tsv"))
    all_s1_ids = s1["entity_id"].tolist()
    print(f"[predict] {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3 test records",
          flush=True)

    print("[predict] generating candidates (blocking)...", flush=True)
    scored_path = os.path.join(args.output_dir, "_predict_candidates_scored.tsv")
    n_pairs, n_s1 = generate_candidates(
        s1, s2, s3, out_path=scored_path, top_k=args.top_k)
    print(f"[predict] {n_pairs} candidate pairs for {n_s1}/{len(all_s1_ids)} S1",
          flush=True)

    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    if n_pairs == 0:
        write_id_list_tsv(candidate_path,
                           ["source1_entity_id", "candidate_entity_ids"],
                           {}, all_s1_ids)
        write_id_list_tsv(os.path.join(args.output_dir, "matching_results.tsv"),
                           ["source1_entity_id", "matched_entity_ids"],
                           {}, all_s1_ids)
        print("[predict] no candidates found; wrote empty outputs", flush=True)
        return

    # Validator-format candidate file (streams; assumes S1-grouped rows).
    write_candidate_id_list(scored_path, candidate_path, all_s1_ids)

    bundle = joblib.load(args.model)
    clf, threshold = bundle["classifier"], bundle["threshold"]
    print(f"[predict] model threshold={threshold:.2f}", flush=True)

    # Cache only referenced entities, then free the big frames.
    print("[predict] building entity caches...", flush=True)
    oids = collect_candidate_oids(scored_path)
    print(f"[predict] unique candidate S2/S3 ids: {len(oids)}", flush=True)
    s1_cache = build_entity_cache(s1, all_s1_ids)
    del s1
    gc.collect()
    ref_cache = build_entity_cache(s2, oids)
    del s2
    gc.collect()
    ref_cache.update(build_entity_cache(s3, oids))
    del s3
    gc.collect()

    print("[predict] scoring pairs...", flush=True)
    match_map = {}
    done = 0
    t0 = time.time()
    for chunk in pd.read_csv(scored_path, sep="\t", chunksize=CHUNK):
        X = build_features(chunk, s1_cache, ref_cache, workers=args.workers)
        probs = clf.predict_proba(X.to_numpy(dtype=np.float32))[:, 1]
        for s1id, oid, p in zip(chunk["source1_entity_id"],
                                chunk["candidate_entity_id"], probs):
            if p >= threshold:
                match_map.setdefault(s1id, set()).add(oid)
        done += len(chunk)
        if done % (CHUNK * 10) < CHUNK:
            print(f"[predict]   scored {done}/{n_pairs} pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    del s1_cache, ref_cache
    gc.collect()

    write_id_list_tsv(
        os.path.join(args.output_dir, "matching_results.tsv"),
        ["source1_entity_id", "matched_entity_ids"],
        match_map,
        all_s1_ids,
    )
    n_matched = sum(1 for v in match_map.values() if v)
    print(f"[predict] wrote matching_results.tsv "
          f"({n_matched}/{len(all_s1_ids)} S1 matched, total {time.time() - t_all:.0f}s)",
          flush=True)
    # The scored temp (tens of GB on full test) has served its purpose.
    try:
        os.remove(scored_path)
        print(f"[predict] removed temp {scored_path}", flush=True)
    except OSError:
        pass


if __name__ == "__main__":
    main()
