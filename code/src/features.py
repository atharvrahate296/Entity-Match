"""Pairwise feature engineering for (Source-1, candidate) pairs.

All features are symmetric string/attribute comparisons — nothing here
touches any external service or data source (fair-play requirement).

Performance design: entity-level normalized strings and postal codes are
precomputed ONCE per entity in ``data_io.build_entity_cache``. Token sets
and each name's first token used to also be precomputed and stored per
entity as frozensets, but that added several hundred bytes of object
overhead per entity — with millions of entities cached at once (S1 sample
or full set, plus every referenced S2/S3 record), that was enough extra
memory to push a 16GB machine into an out-of-memory crash. They're cheap
to recompute with a plain ``str.split()`` right here instead, so the cache
no longer stores them at all -- this trades a small, constant amount of
extra CPU per pair for a large, constant reduction in cache memory.

The per-pair loop below is therefore lean scalar ops + a couple of
str.split() calls + rapidfuzz C++ calls, optionally split across threads
(rapidfuzz releases the GIL, cache entries are read-only).

Length deltas are scaled to ~unit range so the SGD classifier is not
dominated by raw character counts.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from data_io import C_NAME, C_ADDR, C_COUNTRY, C_POSTAL

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
    # Added to help recall: the validation slice showed decent precision
    # (0.87) but weak recall (0.51), meaning the classifier/threshold
    # combination wasn't confident enough on genuine matches with more
    # unusual wording. These target exactly that gap:
    "name_jaro_winkler",   # rewards a shared PREFIX heavily -- good for
                           # typos/transliteration near the start of a name
    "name_partial_ratio",  # best-matching substring score -- catches
                           # "Joe's Pizza" vs "Joe's Pizza Downtown Location"
                           # (DBA/trade names, truncated vs. full names)
    "address_containment", # overlap / smaller side's token count -- catches
                           # a short/partial address that's fully contained
                           # in a longer, more complete one for the same place
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


def _containment(a, b):
    """Overlap divided by the SMALLER side's size, rather than the union
    (Jaccard). This scores 1.0 whenever one token set is fully contained in
    the other, regardless of how much longer the other one is -- useful for
    partial addresses ("12 Main St" contained inside "12 Main St, Suite 4,
    Springfield")."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    smaller = min(len(a), len(b))
    return inter / smaller if smaller else 0.0


def _featurize_range(s1_ids, cand_ids, scores, s1_cache, other_cache, lo, hi):
    out = np.empty((hi - lo, len(FEATURE_COLUMNS)), dtype=np.float32)
    for k, idx in enumerate(range(lo, hi)):
        a = s1_cache[s1_ids[idx]]
        b = other_cache[cand_ids[idx]]
        name_a, name_b = a[C_NAME], b[C_NAME]
        addr_a, addr_b = a[C_ADDR], b[C_ADDR]
        # Recomputed on demand -- see module docstring. str.split() on an
        # already-normalized (lowercased, punctuation-stripped) string is a
        # cheap, fast C-level call.
        name_toks_a = frozenset(name_a.split()) if name_a else frozenset()
        name_toks_b = frozenset(name_b.split()) if name_b else frozenset()
        addr_toks_a = frozenset(addr_a.split()) if addr_a else frozenset()
        addr_toks_b = frozenset(addr_b.split()) if addr_b else frozenset()
        first_a = name_a.split(" ", 1)[0] if name_a else ""
        first_b = name_b.split(" ", 1)[0] if name_b else ""

        out[k, 0] = scores[idx]
        out[k, 1] = _jaccard(name_toks_a, name_toks_b)
        out[k, 2] = fuzz.ratio(name_a, name_b) / 100.0
        out[k, 3] = fuzz.token_sort_ratio(name_a, name_b) / 100.0
        out[k, 4] = abs(len(name_a) - len(name_b)) / NAME_LEN_SCALE
        out[k, 5] = 1.0 if (first_a and first_a == first_b) else 0.0
        out[k, 6] = _jaccard(addr_toks_a, addr_toks_b)
        out[k, 7] = fuzz.ratio(addr_a, addr_b) / 100.0
        out[k, 8] = abs(len(addr_a) - len(addr_b)) / ADDR_LEN_SCALE
        pa, pb = a[C_POSTAL], b[C_POSTAL]
        out[k, 9] = 1.0 if (pa is not None and pa == pb) else 0.0
        out[k, 10] = 1.0 if a[C_COUNTRY] == b[C_COUNTRY] else 0.0
        out[k, 11] = JaroWinkler.normalized_similarity(name_a, name_b)
        out[k, 12] = fuzz.partial_ratio(name_a, name_b) / 100.0
        out[k, 13] = _containment(addr_toks_a, addr_toks_b)
    return out


def build_features(pairs_df, s1_lookup, other_lookup, workers=4):
    """Engineer features for candidate pairs.

    s1_lookup / other_lookup may be the legacy dict-of-dicts (with
    norm_name / norm_address / business_address / country keys) or the
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
    """Convert legacy dict-of-dicts lookups to the compact (norm_name,
    norm_address, country, postal) cache tuples -- matches the current
    data_io.build_entity_cache layout (no stored token sets/first token;
    those are recomputed in _featurize_range)."""
    from normalize import extract_postal_code
    cache = {}
    for eid, d in lookup.items():
        nn, na = d["norm_name"], d["norm_address"]
        cache[eid] = (
            nn, na, d.get("country", ""),
            extract_postal_code(d.get("business_address", "")),
        )
    return cache