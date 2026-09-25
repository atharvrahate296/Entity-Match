"""TSV I/O helpers shared by train.py and predict.py."""

import csv
import os

import pandas as pd

from normalize import normalize_name, normalize_address


def load_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    # `country` repeats a small set of distinct strings across millions of
    # rows -- category dtype stores each value once instead of once per row.
    df["country"] = df["country"].astype("category")
    return df


def load_ground_truth(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    truth_map = {}
    for s1, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        ids = set(x for x in matched.split(",") if x) if matched else set()
        truth_map[s1] = ids
    return truth_map


def build_lookup(df):
    """entity_id -> dict of normalized fields, keyed for feature building.

    If `df` already has norm_name/norm_address columns (i.e. it already went
    through blocking.prepare(), which mutates in place), those are reused
    instead of recomputing normalization a second time over the same
    millions of rows.
    """
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


def candidates_to_pairs_df(candidates):
    """dict s1 -> {other_id: score} -> flat DataFrame.

    Built as three parallel lists (rather than a list of row tuples) and
    handed to pandas as columns directly -- avoids the intermediate
    per-row Python tuple + DataFrame(rows=...) construction, which matters
    once the candidate count reaches the tens of millions.
    """
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
    """
    Write a matching_results.tsv / candidate_pairs.tsv style file.
    id_map: source1_entity_id -> set/iterable of matched or candidate ids.
    required_ids: full ordered list of Source-1 ids that must each get one row.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for s1 in required_ids:
            ids = sorted(set(id_map.get(s1, ())))
            writer.writerow([s1, ",".join(ids)])
