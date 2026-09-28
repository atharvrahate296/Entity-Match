# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Thesis Code
**Team Members:** Atharv Rahate(Team leader), Tanmay Chaudhari, Rohan Kokatare
**Submission Date:** 27/09/2026

---

## 1. Executive Summary

A blocking + classifier pipeline for cross-source business entity resolution,
built and tuned end-to-end on a single 16GB laptop with no GPU acceleration
(no CUDA-based library in this stack has a usable non-CUDA GPU path, and the
actual bottlenecks encountered were algorithmic/memory issues rather than
raw compute — see Section 3 and the Appendix for the iteration history).

Candidates are generated per Source-1 record via an **inverted-index token
blocking** stage over normalized name/address words, plus a second,
independent **exact postal-code blocking key** to recover matches that share
no useful word at all. Candidates are then scored by an `SGDClassifier`
(log-loss, trained via streaming `partial_fit`) on 14 pairwise
string-similarity features, with the decision threshold tuned directly
against the leaderboard's macro F_0.5 metric rather than a generic accuracy
proxy — this matters because F_0.5 punishes false merges twice as hard as
misses, and singletons scored 0 vs 1 on any false merge dominates the metric
in practice.

*[Fill in your final validation F_0.5 score from `score_predictions.py` and
your latest leaderboard score once your most recent retrain has been
validated — see Section 5.]*

---

## 2. Methodology

### 2.1 Problem Analysis

Noise patterns the pipeline is built to absorb (per the problem statement's
"Noise Patterns to Expect"):

- **Name noise:** legal-suffix inconsistency (Corp/Corporation,
  Pvt/Private, Ltd/Limited, and multi-word forms like "limited liability
  company" → "llc"), `&` vs "and", DBA/trade names, word-order
  transpositions, typos, and transliteration/accented characters
  (e.g. "Café" vs "Cafe").
- **Address noise:** street-type abbreviations (Rd/Road, St/Street,
  Sector/Sec, Phase/Ph), transliteration variants, missing components (no
  PIN/ZIP, no state), landmark references ("Near SBI ATM"), municipal
  numbering and component reordering.
- **Open-set `country` field:** training only contains `US`/`India`; the
  test set adds `France`. The pipeline never hard-codes or one-hot-encodes
  a fixed country list — `country` is used only as an equality predicate
  (for blocking and as a feature), which works identically for any string
  value, seen in training or not.
- **Landmark references are actively removed, not just abbreviated.**
  Early iteration kept words like "near"/"opposite" as tokens; this was
  changed because the landmark phrase for the SAME real address is often
  worded completely differently across two source records, so keeping it
  can make a true match look *less* similar rather than more.

*[Add any dataset-specific observations from your own EDA — e.g. exact
field-length distributions, missingness rates, per-country class balance
of matches vs singletons. One data point already measured: on a 20,000
held-out validation sample, only ~5.6% of Source-1 entities were true
singletons — the large majority genuinely require correct matching, so a
trivial "predict nothing" baseline would score far below the leaderboard's
observed top scores.]*

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (supervised pairwise matching),
laptop-constrained (16GB RAM, no GPU use).

**Core Innovation 1 — dual-key blocking.** Word-token inverted-index
blocking alone can never surface a true match that shares no useful word at
all (heavy abbreviation or complete rewording of the name/address). A
second, independent blocking key on exact postal code recovers exactly this
case cheaply, since postal-code buckets are normally small and highly
discriminative.

**Core Innovation 2 — metric-aligned threshold selection.** The classifier's
match-probability threshold is chosen by a vectorized grid search that
directly maximizes macro-averaged F_0.5 on a held-out validation split
(numpy `bincount` over factorized entity codes, not a per-threshold Python
loop over entities), aligning model selection with the actual scoring
metric and its precision/recall asymmetry rather than a generic AUC proxy.

**Core Innovation 3 — memory-bounded design forced by real constraints.**
Every stage streams in bounded chunks and never holds more than one
country's data resident at once (see Section 3 and Appendix C for the
concrete out-of-memory failures this was built to fix). This is a genuine
engineering constraint of the target hardware, not an optimization for its
own sake — earlier full-table/whole-dataset approaches (e.g. TF-IDF cosine
similarity blocking, unioning all countries' candidates before caching)
were tried first and abandoned specifically because they failed on this
hardware.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used** (`src/blocking.py`):

1. **Country bucket** (exact string match) — records are partitioned by the
   `country` field before any similarity work, since a true cross-source
   match is expected to share a country label. Works identically for a
   country absent from training (France, at test time).

2. **Word-token inverted index** (primary key, within each country bucket):
   - Names and addresses are normalized (`src/normalize.py`) and split into
     word tokens; legal-suffix filler words (`inc`, `ltd`, `co`, `and`,
     `the`, ...) and short tokens (< 3 chars) are excluded from indexing.
   - An inverted index (`token → record positions`) is built over the
     country's Source-2/3 references. Any single token's posting list is
     capped at `MAX_POSTING_LIST = 2000` entries during index build, so an
     unusually common word can't make the index unboundedly large.
   - For each Source-1 query, its own tokens' posting lists are visited
     **smallest-first** (rarer tokens are more discriminative and cheaper
     to check) up to a fixed work budget, `MAX_POSTINGS_TOUCHED = 3000`
     posting-list entries — this bounds per-record query cost regardless of
     how many common words a record happens to contain, which is what
     makes this approach genuinely sub-quadratic instead of exhaustive.
   - Of the candidates discovered this way, the top `MAX_SCORE_CANDIDATES
     = 800` by shared-token count get real similarity scoring
     (`rapidfuzz` token-sort ratio on name, ratio on address, blended with
     token overlap); the top `TOP_K = 20` above `MIN_SIM = 0.10` per query
     are kept.

3. **Exact postal-code index** (secondary, independent key): a separate
   `postal code → record positions` index (capped at `POSTAL_BUCKET_MAX =
   3000` per code, guarding only against a placeholder/default postal
   value) is checked for every query in addition to the token index.
   Postal-matched candidates are added to the candidate pool with a scoring
   bonus so they survive the shortlist step even with low token overlap.
   This recovers matches the token index can structurally never find:
   two records for the same business with completely different, heavily
   abbreviated or reworded name/address text, but the same postal code.

**Why this replaced an earlier TF-IDF/cosine approach:** the first
implementation used character 2-4-gram `HashingVectorizer` features and
sparse cosine similarity. Short character n-grams collide across nearly
every business name, so the "sparse" similarity matrices came out nearly
dense — effective cost approached `O(n_S1 × n_ref)` per country, which at
this dataset's scale (millions of records per source) made the blocking
stage run for an unbounded amount of time rather than completing. The
token-index approach's cost instead scales with the sum of block sizes
actually visited, which is what makes it tractable on a single laptop.

- **Candidate pairs generated:** *[fill in the count printed by
  `train.py`/`predict.py` on your most recent run]*
- **Blocking recall ceiling:** *[fill in the value `train.py` prints —
  this is the fraction of ground-truth matches present in the candidate
  set before the classifier ever sees them, i.e. the hard upper bound on
  achievable recall. An earlier, lower-budget configuration measured
  0.6681 on a 120k-entity training sample — a third of true matches were
  structurally unreachable at that budget. If this number is still well
  below your target after the postal-key/budget changes, the next lever
  is raising `MAX_POSTINGS_TOUCHED`/`TOP_K` further, not classifier work,
  since no amount of classifier tuning recovers a candidate that was
  never generated.]*

---

## 4. Matching Model

**Features used** (`src/features.py`, 14 total):

*Name features:*
- `name_tfidf_cosine` — the blocking-stage blended similarity score,
  carried through as a feature (historical name; no longer a literal
  TF-IDF cosine value since the blocking rewrite — see Section 3)
- `name_jaccard` — token-set Jaccard similarity
- `name_levenshtein_ratio` — `rapidfuzz` character-level ratio
- `name_token_sort_ratio` — order-invariant token-sorted ratio
- `name_len_diff` — scaled character-length delta
- `first_token_match` — exact match on the first normalized name token
- `name_jaro_winkler` — prefix-weighted similarity; strong for typos and
  transliteration errors concentrated near the start of a name
- `name_partial_ratio` — best-matching-substring score; catches DBA/trade
  names and truncated-vs-full name pairs (e.g. "Joe's Pizza" vs. "Joe's
  Pizza Downtown Location")

*Address features:*
- `address_jaccard` — token-set Jaccard similarity
- `address_levenshtein_ratio` — `rapidfuzz` character-level ratio
- `address_len_diff` — scaled character-length delta
- `address_containment` — token overlap divided by the *smaller* side's
  token count (not the union); scores high whenever one address is a
  partial/shortened version fully contained in a fuller one

*Other:*
- `postal_match` — exact match on the regex-extracted 4-6 digit postal code
- `country_match` — exact match on the `country` field

**Model type:** `sklearn.linear_model.SGDClassifier` (log-loss, L2 penalty),
trained from scratch via streaming `partial_fit` over batches read from a
scored-candidates file (BSD-licensed, well within the MIT/Apache-2.0,
≤8B-parameter constraint; not a pretrained model at all). This choice was
made specifically for the 16GB-laptop constraint: `partial_fit` needs a
single streaming pass with no full in-memory design matrix, unlike batch
`fit`. Class weights are computed from an exact label-count pass over the
training split (`partial_fit` rejects the string shortcut
`class_weight="balanced"`, so an explicit `{0: w0, 1: w1}` dict is
computed and passed instead).

*[Note for anyone extending this: a linear model can only combine these 14
features additively — it cannot learn interaction effects (e.g. "high name
similarity is much stronger evidence when postal also matches, but weak
evidence when country doesn't match at all"). A gradient-boosted tree
model (e.g. LightGBM or `HistGradientBoostingClassifier`, both well within
the license/size constraint) captures such interactions natively and is a
natural next step if linear-model capacity turns out to be the limiting
factor once blocking recall is no longer the bottleneck.]*

**Threshold selection method:** vectorized grid search (`np.arange(0.05,
0.96, 0.02)`) over the classifier's positive-class probability, evaluated
by macro F_0.5 on a held-out validation split. The search is implemented
with `numpy.bincount` over factorized entity codes rather than a
per-threshold Python loop over entities, so a full sweep over ~46
thresholds across roughly a million validation pairs completes in seconds.
The validation split itself is a group-wise split by Source-1 entity
(stratified by country, no entity's pairs appear on both sides of the
split), so the tuned threshold reflects genuine held-out performance.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), local validation:** *[fill in from
  `score_predictions.py`'s output on `output/validation/matching_results.tsv`
  — confirm the `validate.py` run's `--max-val-s1`/`--val-frac`/`--seed`
  actually match the `train.py` run that produced the model being scored,
  since a mismatch here silently reproduces the wrong held-out set without
  any error message.]*
- **F_0.5 Score, leaderboard:** *[fill in your latest submitted score]*
- **Micro precision / recall:** *[fill in — precision and recall moving in
  opposite directions across pipeline changes was the main diagnostic
  signal used throughout development; see Appendix C]*
- **Common false positives (wrong merges):** distinct businesses sharing an
  address or plaza (shared postal code, similar name tokens from a
  shopping center or office park); generic/common business names with
  weak address discrimination. *[Update with concrete examples once you've
  inspected validation errors directly.]*
- **Common false negatives (missed matches):** heavily abbreviated or
  reworded names/addresses that share too few tokens to be discovered by
  the primary blocking key (the postal-code secondary key targets this
  directly, but doesn't help when postal code itself is also missing or
  inconsistent); landmark-heavy addresses with little else in common.
  *[Update with concrete examples once inspected.]*

---

## 6. Conclusion

The pipeline combines a dual-key (word-token + postal-code) blocking stage
with a supervised linear classifier whose decision threshold is tuned
directly against the actual scoring metric, engineered throughout for a
genuine 16GB-RAM, no-GPU hardware constraint rather than assuming
unlimited compute.

**Lessons learned during development** (see Appendix C for the full
iteration history): an initial TF-IDF/cosine blocking approach that looked
reasonable in isolation failed catastrophically at this dataset's actual
scale, because short character n-grams collide often enough to make
"sparse" similarity computation effectively dense. Out-of-memory failures
during entity caching were resolved by processing one country at a time
and shrinking each cached entity to only the four fields actually used by
the feature functions. A significant, easy-to-miss failure mode was
discovered where two support scripts (`train.py` and `validate.py`) fell
out of sync on a sampling parameter, silently causing the validation
scorer to evaluate the wrong held-out entities without any error —
underscoring that a local scoring pipeline needs the same scrutiny as the
main training pipeline, since a plausible-looking but wrong number is
worse than an obvious crash.

*[Summarize your final results and any additional lessons once you have a
fresh, verified end-to-end run against the real dataset.]*

---

## Appendix

### A. Code Artefacts

Runnable pipeline ships under `code/`:

```
code/
├── README.md              # exact run instructions
├── requirements.txt       # pinned dependencies
├── models/
│   └── model.joblib        # produced by train.py: {classifier, threshold, top_k, seed}
└── src/
    ├── normalize.py         # name/address cleanup: accent stripping, legal-suffix
    │                        # and multi-word phrase collapsing, landmark-word removal,
    │                        # postal-code extraction
    ├── blocking.py           # candidate generation: word-token inverted index +
    │                        # exact postal-code secondary key (see Section 3)
    ├── data_io.py             # TSV loading, low-RAM entity cache, output writers
    ├── features.py            # 14 pairwise features (see Section 4)
    ├── model.py               # SGDClassifier + vectorized macro-F0.5 threshold search
    ├── train.py                # entry point: train on a stratified sample + tune threshold
    ├── predict.py               # entry point: generate output/*.tsv on the real test set
    ├── validate.py               # entry point: reproduce train.py's held-out split and
    │                            # run real inference on it, for local scoring
    └── score_predictions.py      # standalone precision/recall/macro-F0.5 scorer
```

Entry points (run from `student_resource/`, see `README.md` for full details
and current default flag values):

```bash
python code/src/train.py
python code/src/predict.py
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
python code/src/validate.py
python code/src/score_predictions.py \
    --predictions output/validation/matching_results.tsv \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --only-ids output/validation/val_ids.txt
```

### B. Additional Results

*[Add charts/tables here — e.g. F_0.5 vs threshold curve, precision/recall
by country, candidate-set size distribution, blocking recall ceiling before
vs. after the postal-key addition.]*

### C. Development iteration history (technical depth / debugging record)

Included per the challenge's guidance to prioritize clarity and technical
depth: the pipeline went through several distinct, diagnosed failure modes
before reaching its current form, each with a concrete root cause rather
than trial-and-error tuning:

1. **Blocking never completed.** Root cause: character n-gram TF-IDF
   cosine similarity produced near-dense similarity matrices at this
   dataset's scale (effective cost ≈ `O(n_S1 × n_ref)` per country).
   Fixed by replacing it with word-token inverted-index blocking.
2. **Out-of-memory during entity caching.** Two compounding causes: (a)
   candidate ids from all countries were unioned before caching, holding
   the majority of the entire reference universe in memory at once —
   fixed by caching one country at a time; (b) each cached entity stored
   an unused raw address field plus two `frozenset` token sets, adding
   substantial Python object overhead per entity across millions of
   entities — fixed by shrinking the cache to the four fields actually
   read by the feature functions and recomputing token sets on demand.
3. **Invalid local scoring.** Test-set predictions were scored against
   training-set ground truth — two disjoint entity universes with no
   correspondence between identically-formatted IDs — producing a
   meaningless number. Fixed by building `validate.py` to generate
   predictions on a genuine held-out slice of *training* data instead.
4. **Recall-limited by blocking budget.** Diagnosed via `validate.py` +
   `score_predictions.py`: precision was reasonable but recall was well
   below it, pointing at candidates being lost before the classifier ever
   scored them. Addressed by raising blocking's work budgets and adding
   the postal-code secondary blocking key described in Section 3.
5. **Silent train/validate parameter drift.** `train.py`'s
   `--max-val-s1` default was changed without updating `validate.py`'s
   matching default, causing `validate.py` to reproduce a different,
   incorrect held-out split with no error — the local score reported was
   real numbers, just for the wrong entities. Fixed by syncing the
   defaults and adding an explicit warning in `validate.py`'s docstring.

---
