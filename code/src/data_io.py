import csv
import os

import pandas as pd

from normalize import normalize_name, normalize_address


def load_source(path, chunksize=200_000, usecols=None):
    # Chunked read: a single full pd.read_csv on these 200-500MB files
    # intermittently fails on Windows with
    #   ParserError: Calling read(nbytes) on source failed,
    # while chunked reads succeed reliably. Concat at the end.
    parts = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        chunksize=chunksize, usecols=usecols,
    )
    df = pd.concat(parts, ignore_index=True)
    if "country" in df.columns:
        df["country"] = df["country"].astype("category")
    return df


def load_ground_truth(path, chunksize=200_000, keep_ids=None):
    """Stream ground truth into {s1_id: set(matched ids)}.

    keep_ids: optional set of S1 ids to retain (saves RAM when training on
    a sample — the full file has 2.2M rows).
    """
    truth_map = {}
    parts = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize
    )
    for chunk in parts:
        if keep_ids is not None:
            chunk = chunk[chunk["source1_entity_id"].isin(keep_ids)]
            if chunk.empty:
                continue
        for s1, matched in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            ids = set(x for x in matched.split(",") if x) if matched else set()
            if s1 in truth_map:
                truth_map[s1] |= ids
            else:
                truth_map[s1] = ids
    return truth_map


def build_lookup(df):
    has_norm = "norm_name" in df.columns and "norm_address" in df.columns
    lookup = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        lookup[d["entity_id"]] = {
            "norm_name": d["norm_name"] if has_norm else normalize_name(d["business_name"]),
            "norm_address": d["norm_address"] if has_norm else normalize_address(d["business_address"]),
            "business_address": d["business_address"],
            "country": d["country"],
        }
    return lookup


# --- Low-memory entity cache -----------------------------------------------
# Tuple layout (positional, to avoid per-entity dict overhead for millions
# of entities): (norm_name, norm_address, country, postal).
#
# Previously this also stored the raw (un-normalized) business_address and
# precomputed frozenset token sets for name/address. Neither is worth the
# memory at scale: the raw address is never read by features.py (only the
# normalized string and the postal code extracted from it are used), and
# the frozensets cost several hundred bytes each once you count their own
# object overhead plus a separate string object per token. With millions
# of entities cached at once (S1 sample/full set + referenced S2/S3), that
# was enough extra memory to tip a 16GB machine into an out-of-memory
# crash. Token sets and the first-token value are cheap to recompute with
# a plain str.split() right when features.py needs them, so they're no
# longer stored here at all.
C_NAME, C_ADDR, C_COUNTRY, C_POSTAL = 0, 1, 2, 3


def build_entity_cache(df, ids=None):
    """Build a compact per-entity cache: (norm_name, norm_address, country,
    postal_code). Only entities in `ids` are cached (when given).
    """
    from normalize import extract_postal_code

    if ids is not None:
        ids = set(ids)
        df = df[df["entity_id"].isin(ids)]
    has_norm = "norm_name" in df.columns and "norm_address" in df.columns
    cache = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        nn = d["norm_name"] if has_norm else normalize_name(d["business_name"])
        na = d["norm_address"] if has_norm else normalize_address(d["business_address"])
        cache[d["entity_id"]] = (
            nn,
            na,
            d["country"],
            extract_postal_code(d["business_address"]),
        )
    return cache


def build_entity_cache_from_path(path, ids, chunksize=200_000):
    """Stream a source TSV from disk, caching only rows whose id is in `ids`.

    Low-RAM alternative to load_source()+build_entity_cache(): never holds
    the full table, only matching rows (candidate refs are a small fraction
    of the 10M-row tables).
    """
    from normalize import extract_postal_code, normalize_name, normalize_address

    want = set(ids)
    cache = {}
    if not want:
        return cache
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=chunksize):
        chunk = chunk[chunk["entity_id"].isin(want)]
        if chunk.empty:
            continue
        has_norm = "norm_name" in chunk.columns and "norm_address" in chunk.columns
        for row in chunk.itertuples(index=False):
            d = row._asdict()
            eid = d["entity_id"]
            if eid not in want or eid in cache:
                continue
            nn = d["norm_name"] if has_norm else normalize_name(d["business_name"])
            na = d["norm_address"] if has_norm else normalize_address(d["business_address"])
            cache[eid] = (
                nn,
                na,
                d["country"],
                extract_postal_code(d["business_address"]),
            )
            if len(cache) >= len(want):
                break
        del chunk
        if len(cache) >= len(want):
            break
    return cache


def collect_candidate_oids(scored_path, chunksize=200_000):
    """Stream a per-pair scored candidate file, return set of candidate ids."""
    oids = set()
    for chunk in pd.read_csv(scored_path, sep="\t", chunksize=chunksize,
                             usecols=["candidate_entity_id"]):
        oids.update(chunk["candidate_entity_id"].tolist())
    return oids


def write_candidate_id_list(scored_path, out_path, required_ids, chunksize=200_000):
    """Convert per-pair scored file -> validator id-list format.

    Assumes rows are grouped by source1_entity_id (guaranteed by
    blocking.generate_candidates, which flushes per S1). Single streaming
    pass + empty rows for S1 ids with no candidates.
    """
    emitted = set()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(scored_path, encoding="utf-8") as fin, \
            open(out_path, "w", newline="", encoding="utf-8") as fout:
        header = fin.readline()
        assert "source1_entity_id" in header and "candidate_entity_id" in header, \
            f"unexpected scored header: {header!r}"
        writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        cur_s1, cur_set = None, set()
        for line in fin:
            s1, tab, rest = line.partition("\t")
            if not tab:
                continue
            oid = rest.split("\t", 1)[0].strip()
            if s1 != cur_s1:
                if cur_s1 is not None:
                    writer.writerow([cur_s1, ",".join(sorted(cur_set))])
                    emitted.add(cur_s1)
                cur_s1, cur_set = s1, set()
            if oid:
                cur_set.add(oid)
        if cur_s1 is not None:
            writer.writerow([cur_s1, ",".join(sorted(cur_set))])
            emitted.add(cur_s1)
    # Fill S1 ids that had no candidates at all.
    missing = [s1 for s1 in required_ids if s1 not in emitted]
    if missing:
        with open(out_path, "a", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
            for s1 in missing:
                writer.writerow([s1, ""])
    return len(emitted), len(missing)


def candidates_to_pairs_df(candidates):
    s1_col, other_col, score_col = [], [], []
    for s1, others in candidates.items():
        for other_id, score in others.items():
            s1_col.append(s1)
            other_col.append(other_id)
            score_col.append(score)
    return pd.DataFrame({
        "source1_entity_id": s1_col,
        "candidate_entity_id": other_col,
        "block_score": score_col,
    })


def write_id_list_tsv(path, header, id_map, required_ids):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for s1 in required_ids:
            ids = sorted(set(id_map.get(s1, ())))
            writer.writerow([s1, ",".join(ids)])