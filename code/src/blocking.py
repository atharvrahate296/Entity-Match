"""Candidate generation (blocking) — laptop-safe design.

For each Source-1 record, retrieve the top-K nearest Source-2 / Source-3
records *within the same country bucket* using character n-gram TF-IDF
(HashingVectorizer, stateless) + sparse cosine similarity, unioned with a
cheap shared-first-token fallback for rows the vectorizer under-scores.

Scaling properties (this is what changed vs. the naive version):
- Reference texts are normalized + vectorized ONCE per shard and dotted
  against cached query tiles. The old code re-normalized / re-vectorized
  every reference chunk once per S1 batch (thousands of redundant passes).
- No full-table copies per country: everything is index-sliced per tile /
  per shard, so peak RAM stays bounded on a 16GB laptop.
- Rows are flushed per S1 in S1 order, grouped across S2/S3, so the
  downstream id-list converter can stream in a single pass.

Output is a *scored* per-pair TSV
``source1_entity_id \\t candidate_entity_id \\t block_score`` used internally
for feature engineering. Use ``data_io.write_candidate_id_list`` to produce
the validator-format ``candidate_pairs.tsv`` (one row per S1).
"""

import csv
import gc
import heapq
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from normalize import normalize_name, normalize_address, token_set

TOP_K = 10
MIN_SIM = 0.10

S1_TILE = 2000        # queries per sparse dot
REF_SHARD = 20000     # references vectorized per shard
QUERY_SUPERBATCH = 100000  # S1 rows whose query matrices are cached at once
MAX_TOKEN_BLOCK = 200
N_FEATURES = 2 ** 16


def _make_vectorizer():
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def _norm_texts(names, addrs):
    return [f"{normalize_name(n)} {normalize_address(a)}"
            for n, a in zip(names, addrs)]


def _topk_from_sims(sims, k, min_sim):
    """Per-row top-k (col, score) from a CSR similarity matrix."""
    out = []
    sims = sims.tocsr()
    indptr, indices, data = sims.indptr, sims.indices, sims.data
    for i in range(sims.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            out.append([])
            continue
        idx, vals = indices[s:e], data[s:e]
        if len(vals) > k:
            top = np.argpartition(-vals, k - 1)[:k]
            pairs = sorted(zip(idx[top], vals[top]), key=lambda x: -x[1])
        else:
            order = np.argsort(-vals)
            pairs = [(idx[j], vals[j]) for j in order]
        out.append([(int(j), float(v)) for j, v in pairs if v >= min_sim])
    return out


def _bounded_add(heap, score, oid, k):
    """Maintain a per-S1 bounded min-heap of (score, oid), cap size k."""
    if len(heap) < k:
        heapq.heappush(heap, (score, oid))
    elif score > heap[0][0]:
        heapq.heapreplace(heap, (score, oid))


def generate_candidates(s1_df, s2_df, s3_df, out_path="output/candidate_pairs_scored.tsv",
                        top_k=TOP_K, min_sim=MIN_SIM, s1_tile=S1_TILE,
                        ref_shard=REF_SHARD, query_superbatch=QUERY_SUPERBATCH):
    """Write scored per-pair candidates; return (n_pairs, n_s1_with_candidates)."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    vectorizer = _make_vectorizer()
    t0 = time.time()

    s1_countries = s1_df["country"].to_numpy()
    s1_ids_all = s1_df["entity_id"].to_numpy()
    s1_names_all = s1_df["business_name"].to_numpy()
    s1_addrs_all = s1_df["business_address"].to_numpy()

    total_pairs = 0
    s1_seen = set()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "block_score"])

        # Reference tables as numpy columns (sliced by index, never copied whole).
        ref_tabs = []
        for other_df in (s2_df, s3_df):
            if other_df is None or len(other_df) == 0:
                ref_tabs.append(None)
            else:
                ref_tabs.append((other_df["country"].to_numpy(),
                                 other_df["entity_id"].to_numpy(),
                                 other_df["business_name"].to_numpy(),
                                 other_df["business_address"].to_numpy()))

        # Country buckets present on the S1 side (observed only).
        for country in pd.unique(s1_countries):
            s1_pos = np.flatnonzero(s1_countries == country)
            srcs = []
            for (o_c, o_ids, o_nm, o_ad) in ref_tabs:
                if o_c is None:
                    continue
                o_pos = np.flatnonzero(o_c == country)
                if len(o_pos) > 0:
                    srcs.append((o_c, o_ids, o_nm, o_ad, o_pos))
            if not srcs:
                continue
            n_s1 = len(s1_pos)
            n_ref = sum(len(o_pos) for _, _, _, _, o_pos in srcs)
            c_t = time.time()
            print(f"[blocking] country={country} S1={n_s1} refs={n_ref}", flush=True)

            # Normalize S1 texts once per country (reused by all shards).
            s1_texts = _norm_texts(s1_names_all[s1_pos], s1_addrs_all[s1_pos])
            s1_first = np.array([t.split(" ", 1)[0] if t else "" for t in
                                 s1_names_all[s1_pos]], dtype=object)
            s1_norm_names = [normalize_name(n) for n in s1_names_all[s1_pos]]

            for sb_start in range(0, n_s1, query_superbatch):
                sb_end = min(sb_start + query_superbatch, n_s1)
                sb_n = sb_end - sb_start
                # Cache transformed query tiles for this superbatch.
                q_tiles = []
                for t_start in range(sb_start, sb_end, s1_tile):
                    t_end = min(t_start + s1_tile, sb_end)
                    q_tiles.append(vectorizer.transform(s1_texts[t_start:t_end]))
                # Per-source, per-S1 bounded heaps (sources flushed together
                # per S1 so output rows stay S1-grouped across S2/S3).
                heaps_per_src = []
                for si, (o_c, o_ids, o_nm, o_ad, o_pos) in enumerate(srcs):
                    n_r = len(o_pos)
                    heaps = [[] for _ in range(sb_n)]
                    # Stream reference shards (transform each exactly once).
                    for r_start in range(0, n_r, ref_shard):
                        r_end = min(r_start + ref_shard, n_r)
                        r_idx = o_pos[r_start:r_end]
                        r_mat = vectorizer.transform(
                            _norm_texts(o_nm[r_idx], o_ad[r_idx]))
                        r_ids = o_ids[r_idx]
                        for ti, q_mat in enumerate(q_tiles):
                            sims = q_mat.dot(r_mat.T)
                            for li, pairs in enumerate(_topk_from_sims(sims, top_k, min_sim)):
                                h = heaps[(ti * s1_tile) + li]
                                for j, score in pairs:
                                    _bounded_add(h, score, str(r_ids[j]), top_k)
                        del r_mat, sims
                    heaps_per_src.append(heaps)
                    print(f"[blocking]   {country} src{si + 1} refs {n_r}/{n_r} "
                          f"({time.time() - c_t:.0f}s)", flush=True)
                del q_tiles
                gc.collect()

                # Shared-first-token fallback, only for rows the TF-IDF
                # pass left short (bounds cost to stragglers).
                for si, (o_c, o_ids, o_nm, o_ad, o_pos) in enumerate(srcs):
                    heaps = heaps_per_src[si]
                    needy = [i for i, h in enumerate(heaps) if len(h) < top_k]
                    if not needy:
                        continue
                    # Token -> ref positions index (capped per token).
                    tok_index = {}
                    for j in range(len(o_pos)):
                        nm = o_nm[o_pos[j]]
                        ft = str(nm).split(" ", 1)[0] if nm else ""
                        if ft and len(ft) >= 3:
                            lst = tok_index.get(ft)
                            if lst is None:
                                tok_index[ft] = [j]
                            elif len(lst) < MAX_TOKEN_BLOCK:
                                lst.append(j)
                    ref_norm_cache = {}
                    for i in needy:
                        g = sb_start + i
                        ft = s1_first[g]
                        if not ft or len(ft) < 3 or ft not in tok_index:
                            continue
                        s1_toks = token_set(s1_norm_names[g])
                        if not s1_toks:
                            continue
                        have = {oid for _, oid in heaps[i]}
                        for j in tok_index[ft]:
                            if len(heaps[i]) >= 2 * top_k + 5:
                                break
                            oid = str(o_ids[o_pos[j]])
                            if oid in have:
                                continue
                            rn = ref_norm_cache.get(j)
                            if rn is None:
                                rn = normalize_name(o_nm[o_pos[j]])
                                ref_norm_cache[j] = rn
                            if s1_toks & token_set(rn):
                                heaps[i].append((0.0, oid))
                                have.add(oid)
                    del tok_index, ref_norm_cache

                # Flush superbatch in S1 order (keeps file S1-grouped).
                for i in range(sb_n):
                    s1_id = str(s1_ids_all[s1_pos[sb_start + i]])
                    wrote = False
                    for heaps in heaps_per_src:
                        h = heaps[i]
                        if not h:
                            continue
                        for score, oid in sorted(h, key=lambda x: -x[0]):
                            writer.writerow([s1_id, oid, f"{score:.4f}"])
                            total_pairs += 1
                            wrote = True
                    if wrote:
                        s1_seen.add(s1_id)
                del heaps_per_src
                gc.collect()

                print(f"[blocking] country={country} done "
                      f"({time.time() - c_t:.0f}s, total pairs so far {total_pairs})",
                      flush=True)
                del s1_texts, s1_first, s1_norm_names
                gc.collect()

    print(f"[blocking] wrote {total_pairs} pairs for {len(s1_seen)} S1 "
          f"({time.time() - t0:.0f}s) -> {out_path}", flush=True)
    return total_pairs, len(s1_seen)
