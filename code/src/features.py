"""Pairwise feature engineering for (Source-1, candidate) pairs.

All features are symmetric string/attribute comparisons — nothing here
touches any external service or data source (fair-play requirement).

Performance design: entity-level values (normalized strings, token sets,
lengths, first tokens, postal codes) are precomputed ONCE per entity in
``data_io.build_entity_cache``. The old code re-split token sets (4x) and
re-ran the postal regex (2x) for every pair — for millions of pairs that
dominated runtime. The per-pair loop below is therefore lean scalar ops +
rapidfuzz C++ calls, optionally split across threads (rapidfuzz releases
the GIL, cache entries are read-only).

Length deltas are scaled to ~unit range so the SGD classifier is not
dominated by raw character counts.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz

from data_io import (C_NAME, C_ADDR, C_COUNTRY, C_NAME_TOKS, C_ADDR_TOKS,
                      C_FIRST_TOK, C_POSTAL)

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

# Scale raw char-count deltas into ~unit range for SGD.
NAME_LEN_SCALE = 20.0
ADDR_LEN_SCALE = 40.0


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _featurize_range(s1_ids, cand_ids, scores, s1_cache, other_cache, lo, hi):
    out = np.empty((hi - lo, len(FEATURE_COLUMNS)), dtype=np.float32)
    for k, idx in enumerate(range(lo, hi)):
        a = s1_cache[s1_ids[idx]]
        b = other_cache[cand_ids[idx]]
        name_a, name_b = a[C_NAME], b[C_NAME]
        addr_a, addr_b = a[C_ADDR], b[C_ADDR]
        out[k, 0] = scores[idx]
        out[k, 1] = _jaccard(a[C_NAME_TOKS], b[C_NAME_TOKS])
        out[k, 2] = fuzz.ratio(name_a, name_b) / 100.0
        out[k, 3] = fuzz.token_sort_ratio(name_a, name_b) / 100.0
        out[k, 4] = abs(len(name_a) - len(name_b)) / NAME_LEN_SCALE
        out[k, 5] = 1.0 if (a[C_FIRST_TOK] and a[C_FIRST_TOK] == b[C_FIRST_TOK]) else 0.0
        out[k, 6] = _jaccard(a[C_ADDR_TOKS], b[C_ADDR_TOKS])
        out[k, 7] = fuzz.ratio(addr_a, addr_b) / 100.0
        out[k, 8] = abs(len(addr_a) - len(addr_b)) / ADDR_LEN_SCALE
        pa, pb = a[C_POSTAL], b[C_POSTAL]
        out[k, 9] = 1.0 if (pa is not None and pa == pb) else 0.0
        out[k, 10] = 1.0 if a[C_COUNTRY] == b[C_COUNTRY] else 0.0
    return out


def build_features(pairs_df, s1_lookup, other_lookup, workers=4):
    """Engineer features for candidate pairs.

    s1_lookup / other_lookup may be the legacy dict-of-dicts (with
    norm_name / norm_address / business_address / country keys) or the new
    compact cache tuples from ``data_io.build_entity_cache``; legacy dicts
    are adapted on the fly.
    """
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS).astype(np.float32)
    if isinstance(next(iter(s1_lookup.values())), dict):
        s1_lookup = _adapt_legacy(s1_lookup)
        other_lookup = _adapt_legacy(other_lookup)
    s1_ids = pairs_df["source1_entity_id"].to_numpy()
    cand_ids = pairs_df["candidate_entity_id"].to_numpy()
    scores = pairs_df["block_score"].to_numpy(dtype=np.float32)
    n = len(pairs_df)
    if workers is None or workers < 1:
        workers = 1
    workers = min(workers, 8, max(1, n // 2000 or 1))
    if workers == 1:
        mat = _featurize_range(s1_ids, cand_ids, scores, s1_lookup, other_lookup, 0, n)
    else:
        bounds = np.linspace(0, n, workers + 1, dtype=int)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(
                lambda b: _featurize_range(s1_ids, cand_ids, scores, s1_lookup,
                                           other_lookup, b[0], b[1]),
                zip(bounds[:-1], bounds[1:])))
        mat = np.vstack(parts)
    return pd.DataFrame(mat, columns=FEATURE_COLUMNS)


def _adapt_legacy(lookup):
    """Convert legacy dict-of-dicts lookups to compact cache tuples."""
    from normalize import token_set, extract_postal_code
    cache = {}
    for eid, d in lookup.items():
        nn, na = d["norm_name"], d["norm_address"]
        cache[eid] = (
            nn, na, d.get("business_address", ""), d.get("country", ""),
            frozenset(token_set(nn)), frozenset(token_set(na)),
            nn.split(" ", 1)[0] if nn else "",
            extract_postal_code(d.get("business_address", "")),
        )
    return cache
