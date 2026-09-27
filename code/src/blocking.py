"""Candidate generation via inverted-index token blocking.

WHY THIS REPLACES THE OLD APPROACH
-----------------------------------
The previous version used a HashingVectorizer over character 2-4-grams and
computed cosine similarity between every S1 query and every S2/S3 reference
in the same country via sparse matrix dot products. Short character n-grams
("in", "co", "th"...) appear in nearly every business name/address, so even
though the vectors are technically sparse, the resulting query x reference
similarity matrices come out nearly DENSE. That makes the true cost close
to O(n_S1 x n_ref) per country -- with S1/S2/S3 in the millions, that's
hundreds of billions of scored pairs, which is why the pipeline appeared to
freeze rather than genuinely finish or crash.

This version builds a per-country inverted index (word token -> record
positions) over the S2/S3 references, and only ever scores a S1 record
against references that share at least one token with it (classic "token
blocking" from the entity-resolution literature). Cost scales with the sum
of block sizes, not the full cross product. No process pools, no GPU/CUDA
needed -- rapidfuzz (a fast C++ string-similarity library) releases the
GIL, so a small thread pool is enough to use multiple cores for the actual
scoring step.

FAIR-PLAY NOTE: everything here is a comparison of the provided fields
against each other -- no external data, lookups, or services.
"""

import csv
import gc
import heapq
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from normalize import normalize_name, normalize_address, extract_postal_code

TOP_K = 20  # was 15 -- raised again alongside the recall-ceiling fix below;
             # more surviving candidates per record means more chances for
             # the true match to actually be in candidate_pairs.tsv.
MIN_SIM = 0.10  # blended score is in [0, 1]; below this isn't worth keeping

MIN_TOKEN_LEN = 3
MAX_TOKENS_PER_RECORD = 20     # cap tokens indexed/queried per record
MAX_POSTING_LIST = 2000        # was 500 -- a true match sharing only a common
                                # word could previously be truncated OUT of
                                # the index entirely (before any query even
                                # runs) once that word's list hit 500. Raised
                                # so far fewer words get their postings cut.
MAX_SCORE_CANDIDATES = 800     # was 500 -- of the union of postings, only the
                                # top-N by shared-token count get the (more
                                # expensive) rapidfuzz scoring pass
MAX_POSTINGS_TOUCHED = 3000    # was 1200 -- hard budget on posting-list entries
                                # visited per S1 record during the QUERY step.
                                # Measured blocking recall ceiling on the
                                # 120k-S1 training sample was only 0.6681 at
                                # the old budget -- a third of true matches
                                # never became candidates at all. Raised
                                # ~2.5x; expect blocking runtime to scale
                                # roughly the same amount. A record's own
                                # tokens are tried smallest (= most
                                # discriminative) list first, so this budget
                                # is spent on the most useful tokens and
                                # skips the rest once exhausted --
                                # this is what keeps per-record cost bounded
                                # regardless of how many common words a
                                # record happens to contain.
POSTAL_BUCKET_MAX = 3000       # cap on a SECOND, independent blocking key:
                                # exact postal-code match. Word-token
                                # blocking can never find a true match that
                                # shares no useful word at all (heavily
                                # abbreviated/reworded name+address) -- no
                                # budget size fixes that, since the candidate
                                # is never discoverable through tokens in the
                                # first place. Postal code is a separate,
                                # highly discriminative, cheap-to-look-up
                                # signal that recovers exactly this case. The
                                # cap only guards against a placeholder/
                                # default postal value shared by an unusually
                                # large number of records.
PROGRESS_EVERY = 20_000        # print progress every N S1 records scored

# Legal-form / filler words: appear on huge fractions of business records
# and would otherwise create postings lists spanning most of a country.
STOP_TOKENS = {
    "inc", "ltd", "llc", "llp", "corp", "co", "pvt", "and", "the", "of",
    "in", "a", "an", "company", "limited",
}

DEFAULT_N_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))


def _tokens_for_record(norm_name, norm_addr):
    """Blocking-key tokens for one record: name + address word tokens,
    minus stopwords/too-short tokens, capped so no single record dominates
    index-build or query time."""
    toks = []
    for t in norm_name.split():
        if len(t) >= MIN_TOKEN_LEN and t not in STOP_TOKENS:
            toks.append(("n", t))
    for t in norm_addr.split():
        if len(t) >= MIN_TOKEN_LEN and t not in STOP_TOKENS:
            toks.append(("a", t))
    if len(toks) > MAX_TOKENS_PER_RECORD:
        toks = toks[:MAX_TOKENS_PER_RECORD]
    return toks


def _build_inverted_index(ids, names, addrs):
    """token -> [positions] (word-token index), plus postal -> [positions]
    (a second, independent blocking key -- see POSTAL_BUCKET_MAX above for
    why). Also returns normalized name/address arrays (reused later for
    scoring, so we never re-normalize a reference)."""
    n = len(ids)
    norm_names = np.empty(n, dtype=object)
    norm_addrs = np.empty(n, dtype=object)
    index = defaultdict(list)
    postal_index = defaultdict(list)
    for i in range(n):
        nn = normalize_name(names[i])
        na = normalize_address(addrs[i])
        norm_names[i] = nn
        norm_addrs[i] = na
        for tok in _tokens_for_record(nn, na):
            lst = index[tok]
            if len(lst) < MAX_POSTING_LIST:
                lst.append(i)
        postal = extract_postal_code(addrs[i])
        if postal is not None:
            plst = postal_index[postal]
            if len(plst) < POSTAL_BUCKET_MAX:
                plst.append(i)
    return index, postal_index, norm_names, norm_addrs


def _load_ref_country(path, country, chunksize=200_000):
    """Chunked load of a single country's refs (low-RAM: never holds the
    full multi-million-row table at once)."""
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=chunksize):
        m = chunk["country"] == str(country)
        if m.any():
            parts.append(chunk[m])
        del chunk
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    del parts
    return df


def generate_candidates(s1_df, s2_df, s3_df, out_path="output/candidate_pairs_scored.tsv",
                        top_k=TOP_K, min_sim=MIN_SIM, n_workers=None, **_ignored):
    """Write scored per-pair candidates; return (n_pairs, n_s1_with_candidates).

    s2_df / s3_df may be DataFrames (legacy, whole table preloaded) or
    filesystem paths (str/PathLike -- LOW-MEM mode: each country's refs are
    streamed from disk per country, so the parent never holds the full
    ~10M-row tables; peak is one country's refs at a time).

    Same signature/behavior contract as before: output rows are grouped
    contiguously by source1_entity_id (required by
    data_io.write_candidate_id_list), and unused kwargs from the old
    tiling-based version (s1_tile, ref_shard, query_superbatch,
    blocking-workers-as-processes) are accepted and ignored so existing
    train.py/predict.py call sites don't need to change.
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    t0 = time.time()
    n_workers = DEFAULT_N_WORKERS if n_workers is None else max(1, n_workers)
    print(f"[blocking] token-index blocking, {n_workers} scoring thread(s)", flush=True)

    s1_countries = s1_df["country"].to_numpy()
    s1_ids_all = s1_df["entity_id"].to_numpy()
    s1_names_all = s1_df["business_name"].to_numpy()
    s1_addrs_all = s1_df["business_address"].to_numpy()

    total_pairs = 0
    s1_seen = set()

    import pathlib as _pl
    ref_paths, ref_tabs = [], []
    for other in (s2_df, s3_df):
        if isinstance(other, (str, _pl.Path)):
            ref_paths.append(str(other)); ref_tabs.append(None)
        elif other is None or len(other) == 0:
            ref_paths.append(None); ref_tabs.append(None)
        else:
            ref_paths.append(None)
            ref_tabs.append((other["country"].to_numpy(), other["entity_id"].to_numpy(),
                             other["business_name"].to_numpy(), other["business_address"].to_numpy()))
    lowmem = any(p is not None for p in ref_paths)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "block_score"])

        for country in pd.unique(s1_countries):
            s1_pos = np.flatnonzero(s1_countries == country)
            n_s1 = len(s1_pos)
            c_t = time.time()

            ref_ids_parts, ref_names_parts, ref_addrs_parts = [], [], []
            if lowmem:
                for p in ref_paths:
                    if p is None:
                        continue
                    df_c = _load_ref_country(p, country)
                    if df_c is None or len(df_c) == 0:
                        continue
                    ref_ids_parts.append(df_c["entity_id"].to_numpy())
                    ref_names_parts.append(df_c["business_name"].to_numpy())
                    ref_addrs_parts.append(df_c["business_address"].to_numpy())
                    del df_c
                    gc.collect()
            else:
                for tab in ref_tabs:
                    if tab is None:
                        continue
                    o_c, o_ids, o_nm, o_ad = tab
                    m = o_c == country
                    if m.any():
                        ref_ids_parts.append(o_ids[m])
                        ref_names_parts.append(o_nm[m])
                        ref_addrs_parts.append(o_ad[m])

            if not ref_ids_parts:
                print(f"[blocking] country={country} S1={n_s1} refs=0 "
                      f"(no reference records for this country -- all singletons)",
                      flush=True)
                continue

            ref_ids = np.concatenate(ref_ids_parts)
            ref_names = np.concatenate(ref_names_parts)
            ref_addrs = np.concatenate(ref_addrs_parts)
            del ref_ids_parts, ref_names_parts, ref_addrs_parts
            n_ref = len(ref_ids)
            print(f"[blocking] country={country} S1={n_s1} refs={n_ref}", flush=True)

            index, postal_index, ref_norm_names, ref_norm_addrs = _build_inverted_index(
                ref_ids, ref_names, ref_addrs)
            print(f"[blocking]   index built: {len(index)} tokens, "
                  f"{len(postal_index)} postal codes "
                  f"({time.time() - c_t:.0f}s)", flush=True)

            progress_lock = threading.Lock()
            progress_count = [0]

            def _process_range(lo, hi):
                rows = []
                for i in range(lo, hi):
                    g = s1_pos[i]
                    s1_id = str(s1_ids_all[g])
                    nn = normalize_name(s1_names_all[g])
                    na = normalize_address(s1_addrs_all[g])
                    toks = _tokens_for_record(nn, na)
                    own_postal = extract_postal_code(s1_addrs_all[g])
                    if not toks and own_postal is None:
                        rows.append((s1_id, []))
                    else:
                        # Look up each token's posting list, then spend the
                        # touch-budget on the SMALLEST lists first -- small
                        # lists are both cheap and the most discriminative
                        # (a word only 4 other records have is a much
                        # stronger match signal than one 300 records have).
                        # This bounds per-record cost to MAX_POSTINGS_TOUCHED
                        # regardless of how many common words the record's
                        # name/address happen to contain.
                        tok_lists = []
                        for tok in toks:
                            lst = index.get(tok)
                            if lst:
                                tok_lists.append(lst)
                        tok_lists.sort(key=len)
                        counts = {}
                        touched = 0
                        for lst in tok_lists:
                            if touched >= MAX_POSTINGS_TOUCHED:
                                break
                            for pos in lst:
                                counts[pos] = counts.get(pos, 0) + 1
                            touched += len(lst)
                        # Second, independent blocking key: exact postal
                        # match. Added on top of the token-budget candidates
                        # (not subject to MAX_POSTINGS_TOUCHED) -- these
                        # buckets are normally small, and a record that
                        # shares NO useful word token with its true match
                        # (heavy abbreviation/rewording) can only ever be
                        # found this way, regardless of token budget size.
                        # Weighted with a bonus so postal-matched candidates
                        # aren't immediately dropped by the shared-token
                        # shortlist step below even if they share few words.
                        if own_postal is not None:
                            for pos in postal_index.get(own_postal, ()):
                                counts[pos] = counts.get(pos, 0) + 3
                        if not counts:
                            rows.append((s1_id, []))
                        else:
                            if len(counts) > MAX_SCORE_CANDIDATES:
                                shortlist = heapq.nlargest(
                                    MAX_SCORE_CANDIDATES, counts.items(), key=lambda kv: kv[1])
                            else:
                                shortlist = list(counts.items())
                            n_q_toks = len(toks) if toks else 1
                            scored = []
                            for pos, cnt in shortlist:
                                rn, ra = ref_norm_names[pos], ref_norm_addrs[pos]
                                name_score = fuzz.token_sort_ratio(nn, rn) / 100.0
                                addr_score = fuzz.ratio(na, ra) / 100.0
                                tok_overlap = min(cnt / n_q_toks, 1.0)
                                score = 0.5 * name_score + 0.3 * addr_score + 0.2 * tok_overlap
                                if score >= min_sim:
                                    scored.append((score, str(ref_ids[pos])))
                            scored.sort(key=lambda x: -x[0])
                            rows.append((s1_id, scored[:top_k]))
                    with progress_lock:
                        progress_count[0] += 1
                        if progress_count[0] % PROGRESS_EVERY == 0:
                            print(f"[blocking]   {country} scored "
                                  f"{progress_count[0]}/{n_s1} S1 "
                                  f"({time.time() - c_t:.0f}s)", flush=True)
                return rows

            if n_workers > 1 and n_s1 > 2000:
                bounds = np.linspace(0, n_s1, n_workers + 1, dtype=int)
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    chunk_results = list(ex.map(
                        lambda b: _process_range(b[0], b[1]),
                        zip(bounds[:-1], bounds[1:])))
            else:
                chunk_results = [_process_range(0, n_s1)]

            for rows in chunk_results:
                for s1_id, matches in rows:
                    if matches:
                        for score, oid in matches:
                            writer.writerow([s1_id, oid, f"{score:.4f}"])
                            total_pairs += 1
                        s1_seen.add(s1_id)

            del index, postal_index, ref_ids, ref_names, ref_addrs, ref_norm_names, ref_norm_addrs
            gc.collect()
            print(f"[blocking] country={country} done "
                  f"({time.time() - c_t:.0f}s, total pairs so far {total_pairs})",
                  flush=True)

    print(f"[blocking] wrote {total_pairs} pairs for {len(s1_seen)} S1 "
          f"({time.time() - t0:.0f}s) -> {out_path}", flush=True)
    return total_pairs, len(s1_seen)