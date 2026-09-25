"""
Pairwise feature engineering for (Source-1, candidate) pairs.

All features are symmetric string/attribute comparisons — nothing here
touches any external service or data source (fair-play requirement).
"""

import pandas as pd
from rapidfuzz import fuzz

from normalize import extract_postal_code, token_set

FEATURE_COLUMNS = [
    "name_tfidf_cosine",
    "name_jaccard",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_len_diff",
    "first_token_match",
    "address_jaccard",
    "address_levenshtein_ratio",
    "address_len_diff",
    "postal_match",
    "country_match",
]


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_features(pairs_df, s1_lookup, other_lookup):
    """
    pairs_df: DataFrame with columns [source1_entity_id, candidate_entity_id, block_score]
    s1_lookup / other_lookup: dict entity_id -> row (dict) with norm_name, norm_address,
                              business_address (raw), country
    Returns a DataFrame of engineered features aligned with pairs_df's row order.
    """
    rows = []
    for s1_id, other_id, block_score in zip(
        pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"], pairs_df["block_score"]
    ):
        a = s1_lookup[s1_id]
        b = other_lookup[other_id]

        name_a, name_b = a["norm_name"], b["norm_name"]
        addr_a, addr_b = a["norm_address"], b["norm_address"]
        tok_a, tok_b = token_set(name_a), token_set(name_b)
        atok_a, atok_b = token_set(addr_a), token_set(addr_b)

        pin_a = extract_postal_code(a["business_address"])
        pin_b = extract_postal_code(b["business_address"])

        rows.append({
            "name_tfidf_cosine": block_score,
            "name_jaccard": _jaccard(tok_a, tok_b),
            "name_levenshtein_ratio": fuzz.ratio(name_a, name_b) / 100.0,
            "name_token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b) / 100.0,
            "name_len_diff": abs(len(name_a) - len(name_b)),
            "first_token_match": float(bool(name_a) and bool(name_b) and
                                        name_a.split()[0] == name_b.split()[0]) if name_a and name_b else 0.0,
            "address_jaccard": _jaccard(atok_a, atok_b),
            "address_levenshtein_ratio": fuzz.ratio(addr_a, addr_b) / 100.0,
            "address_len_diff": abs(len(addr_a) - len(addr_b)),
            "postal_match": float(pin_a is not None and pin_a == pin_b),
            "country_match": float(a["country"] == b["country"]),
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
