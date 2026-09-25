"""
Candidate generation (blocking).

Strategy
---------
1. Bucket by `country` (an open string label — works for any value, not just
   {US, India}; France or anything else at test time is handled identically).
   Two records that name the same real-world business will carry the same
   country label, so this cuts the search space by roughly the number of
   distinct country values without assuming what those values are.

2. Within a country bucket, vectorize
   `normalized_name + " " + normalized_address`
   with a character n-gram HashingVectorizer (no vocabulary to store) and
   take the top-K nearest Source-2/Source-3 records per Source-1 record by
   cosine similarity.

3. Union in a cheap token-overlap fallback block (shared first-name-token,
   computed lazily) so near-duplicate short names with low n-gram cosine
   still get a chance.

Memory design (important at full scale — millions of rows per source)
-----------------------------------------------------------------------
A country bucket can itself hold hundreds of thousands to millions of rows.
Calling `vectorizer.transform()` ONCE on an entire bucket is what crashes:
`HashingVectorizer` consumes the input as a generator and grows its internal
index/data arrays via repeated `numpy.concatenate`-based resizes, which
transiently needs roughly double the final array size, on top of whatever
the rest of the process (the source DataFrames) already holds. On a
multi-million-row bucket that spike is what triggers the MemoryError, even
though the failing allocation looks tiny in isolation — the process is
already near its ceiling.

So neither side is ever transformed in one call:
- The Source-1 side is processed in small batches (`S1_BATCH` rows at a
  time) — only one batch's query matrix is ever resident.
- The Source-2/3 side is *streamed* in small chunks (`REF_CHUNK` rows at a
  time) and never fully materialized as one matrix — only one chunk is
  resident, discarded after each dot product.
- Only a bounded top-K-per-row candidate dict accumulates across chunks,
  which is small regardless of source size.
- Large intermediates are explicitly `del`ed and garbage-collected at the
  end of each country/source iteration so memory is returned promptly
  rather than held until the whole function returns.

If you still hit memory errors, lower `S1_BATCH` / `REF_CHUNK` further
(e.g. to 1_000) — smaller batches trade some CPU time (more, smaller matrix
multiplies) for lower peak memory. Output is the *last* stage before the
matching model scores pairs, per the challenge's definition of
candidate_pairs.tsv.
"""

import gc
import csv
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from normalize import normalize_name, normalize_address, token_set

TOP_K = 15
MIN_SIM = 0.10

# Tune these down (e.g. 1_000) if you still see MemoryError at your scale;
# tune them up for more speed if you have RAM to spare.
S1_BATCH = 500    # Source-1 rows transformed/queried at a time
REF_CHUNK = 500    # Source-2/3 rows transformed/streamed at a time

# A first-name-token shared by more than this many records in one country
# bucket is too generic to be a useful blocking key (e.g. "the", "national",
# "metro") and is dropped from the fallback index entirely. Without this cap,
# a single common token can fan out into millions of candidate-dict
# insertions (rows_sharing_token x rows_sharing_token) and exhaust memory
# even when the vectorizer-based blocking above is perfectly bounded.
MAX_TOKEN_BLOCK = 200

# Fixed feature space avoids the huge vocabulary TfidfVectorizer would build.
N_FEATURES = 2 ** 16  # further reduced to lower memory per matrix


def _make_vectorizer():
    """HashingVectorizer is stateless (no vocabulary/fit), so a single
    instance can safely be reused across every batch and chunk."""
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def _combined_text(texts_name, texts_addr):
    return [f"{n} {a}" for n, a in zip(texts_name, texts_addr)]


def _write_candidate_rows(f, s1_id, others):
    """Append candidate rows for one S1 entity to the TSV file."""
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    for other_id, score in others.items():
        writer.writerow([s1_id, other_id, score])


def generate_candidates(s1_df, s2_df, s3_df, out_path="candidate_pairs.tsv"):
    """
    Writes candidate pairs TSV to out_path.

    Peak memory is bounded by S1_BATCH x REF_CHUNK regardless of total source size.
    After each country bucket, the candidate dict is flushed to disk and
    reset, so memory does not accumulate across countries.
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    first_run = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        if first_run:
            f.write("source1_entity_id\tcandidate_entity_id\tblock_score\n")

        vectorizer = _make_vectorizer()
        candidates = {}

        for other_df, other_source in [(s2_df, "S2"), (s3_df, "S3")]:
            if other_df is None or len(other_df) == 0:
                continue

            print(f"[blocking] processing {other_source} ({len(other_df)} rows)")

            for country, s1_group in s1_df.groupby("country", observed=True):
                other_group = other_df[other_df["country"] == country]
                if len(other_group) == 0:
                    continue

                other_ids = other_group["entity_id"].tolist()

                # Build first-token index for fallback (only tokens within MAX_TOKEN_BLOCK)
                other_first_tok = [
                    n.split(" ", 1)[0] if n else ""
                    for n in other_group["business_name"].tolist()
                ]

                first_tok_index = {}
                for j, ft in enumerate(other_first_tok):
                    if ft:
                        first_tok_index.setdefault(ft, []).append(j)
                first_tok_index = {
                    ft: idxs for ft, idxs in first_tok_index.items()
                    if len(idxs) <= MAX_TOKEN_BLOCK
                }

                n_s1 = len(s1_group)

                # Process Source-1 in batches within this country bucket
                for s1_start in range(0, n_s1, S1_BATCH):
                    s1_end = min(s1_start + S1_BATCH, n_s1)

                    # Normalize current S1 batch
                    s1_indices = range(s1_start, s1_end)
                    s1_names = []
                    s1_addrs = []
                    for i in s1_indices:
                        row = s1_df.iloc[i]
                        s1_names.append(normalize_name(row["business_name"]))
                        s1_addrs.append(normalize_address(row["business_address"]))

                    batch_ids = s1_group.iloc[s1_start:s1_end]["entity_id"].tolist()
                    batch_texts = _combined_text(s1_names, s1_addrs)

                    q_mat = vectorizer.transform(batch_texts)

                    # Stream the reference side in bounded-size chunks
                    for r_start in range(0, len(other_group), REF_CHUNK):
                        r_end = min(r_start + REF_CHUNK, len(other_group))
                        ref_chunk = other_group.iloc[r_start:r_end]
                        ref_ids = ref_chunk["entity_id"].tolist()

                        # Normalize reference chunk
                        ref_names = []
                        ref_addrs = []
                        for _, row in ref_chunk.iterrows():
                            ref_names.append(normalize_name(row["business_name"]))
                            ref_addrs.append(normalize_address(row["business_address"]))

                        ref_texts = _combined_text(ref_names, ref_addrs)
                        r_mat = vectorizer.transform(ref_texts)

                        sims = q_mat.dot(r_mat.T)  # rows are L2-normed -> cosine
                        top = _topk_from_sims(sims, TOP_K)

                        # Accumulate candidates for this chunk
                        for i, s1_id in enumerate(batch_ids):
                            if s1_id not in candidates:
                                candidates[s1_id] = {}
                            bucket = candidates[s1_id]
                            for j, score in top[i]:
                                oid = ref_ids[j]
                                bucket[oid] = max(bucket.get(oid, 0.0), float(score))

                        del r_mat, sims

                    # Fallback: shared first name token, checked lazily
                    for local_i, s1_id in enumerate(batch_ids):
                        global_i = s1_start + local_i
                        raw_name = s1_df.iloc[global_i]["business_name"]
                        ft = raw_name.split(" ", 1)[0] if raw_name else ""
                        if not ft or len(ft) < 3:
                            continue
                        s1_toks = token_set(normalize_name(raw_name))
                        bucket = candidates.get(s1_id, {})
                        for j in first_tok_index.get(ft, []):
                            oid = other_ids[j]
                            if oid in bucket:
                                continue
                            other_name = normalize_name(other_group.iloc[j]["business_name"])
                            if s1_toks & token_set(other_name):
                                bucket[oid] = 0.0

                    del q_mat

                    # Write accumulated candidates for this batch to TSV and clear dict
                    for s1_id in batch_ids:
                        bucket = candidates.pop(s1_id, {})
                        if bucket:
                            _write_candidate_rows(f, s1_id, bucket)

                # Clear any remaining candidates for this country/other_source combo
                candidates.clear()

                del first_tok_index, other_ids
                gc.collect()

    print(f"[blocking] finished writing to {out_path}")


def _topk_from_sims(sims, k):
    """sims: CSR matrix (n_query x n_ref_chunk). Returns per-row [(col, score), ...]
    filtered to MIN_SIM and capped at k, for this chunk only."""
    out = []
    sims = sims.tocsr()
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz == 0:
            out.append([])
            continue
        idx = row.indices
        data = row.data
        if len(data) > k:
            top = np.argpartition(-data, k - 1)[:k]
        else:
            top = np.arange(len(data))
        pairs = sorted(zip(idx[top], data[top]), key=lambda x: -x[1])
        out.append([(j, s) for j, s in pairs if s >= MIN_SIM])
    return out