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
from data_io import (build_entity_cache, build_entity_cache_from_path,
                     load_source, write_candidate_id_list, write_id_list_tsv)
from features import build_features

CHUNK = 50_000
CACHE_CHUNK = 100_000  # smaller read chunksize while building ref caches --
                        # cheap extra safety margin against the parser's own
                        # buffer allocation, on top of the real fix below.


def _country_candidate_oids(scored_path, s1_ids_this_country, chunksize=CHUNK):
    """Unique candidate ids referenced by ONLY this country's S1 rows.

    Streams the scored-pairs file once, restricted to one country's id set,
    instead of collecting candidate ids across all countries at once -- that
    union is what pushed the previous crash to ~6M ids (most of the whole
    S2+S3 universe) instead of roughly a third of that per country.
    """
    oids = set()
    for chunk in pd.read_csv(scored_path, sep="\t", chunksize=chunksize,
                             usecols=["source1_entity_id", "candidate_entity_id"]):
        m = chunk["source1_entity_id"].isin(s1_ids_this_country)
        if m.any():
            oids.update(chunk.loc[m, "candidate_entity_id"].tolist())
    return oids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model", default="code/models/model.joblib")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Must be >= the training --top-k for consistent recall. "
                         "Must match blocking.py's module-level TOP_K and "
                         "train.py's/validate.py's --top-k.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--blocking-workers", type=int, default=None,
                    help="Unused by the current token-index blocking.py (kept "
                         "for CLI compatibility with older invocations).")
    args = ap.parse_args()
    t_all = time.time()

    s1 = load_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")
    all_s1_ids = s1["entity_id"].tolist()
    # Captured BEFORE we delete s1 below -- this is what lets us process
    # caching/scoring one country at a time instead of all countries' worth
    # of candidates in memory simultaneously.
    s1_ids_by_country = {}
    for sid, c in zip(all_s1_ids, s1["country"].astype(str).tolist()):
        s1_ids_by_country.setdefault(c, set()).add(str(sid))
    print(f"[predict] {len(s1)} S1 test records "
          f"(S2/S3 stream from disk per country, low-RAM)", flush=True)

    print("[predict] generating candidates (blocking)...", flush=True)
    scored_path = os.path.join(args.output_dir, "_predict_candidates_scored.tsv")
    n_pairs, n_s1 = generate_candidates(
        s1, s2_path, s3_path, out_path=scored_path, top_k=args.top_k,
        n_workers=args.blocking_workers)
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

    # S1 cache covers ALL of S1 up front -- this stayed well within RAM even
    # at 1.73M records in your run, so it's not restructured here.
    s1_cache = build_entity_cache(s1, all_s1_ids)
    del s1
    gc.collect()

    match_map = {}
    t0 = time.time()
    done_total = 0
    for country, s1_ids_this in s1_ids_by_country.items():
        c_t = time.time()
        print(f"[predict] {country}: collecting candidate ids...", flush=True)
        oids_this = _country_candidate_oids(scored_path, s1_ids_this)
        print(f"[predict] {country}: unique candidate ids: {len(oids_this)}",
              flush=True)

        ref_cache = build_entity_cache_from_path(s2_path, oids_this, chunksize=CACHE_CHUNK)
        gc.collect()
        ref_cache.update(build_entity_cache_from_path(s3_path, oids_this, chunksize=CACHE_CHUNK))
        gc.collect()
        del oids_this
        print(f"[predict] {country}: ref cache built "
              f"({len(ref_cache)} entities, {time.time() - c_t:.0f}s)", flush=True)

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
            done_total += len(chunk)
            if done_total % (CHUNK * 10) < CHUNK:
                print(f"[predict]   scored {done_total}/{n_pairs} pairs total "
                      f"({time.time() - t0:.0f}s)", flush=True)

        del ref_cache
        gc.collect()
        print(f"[predict] {country}: done ({time.time() - c_t:.0f}s)", flush=True)

    del s1_cache
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