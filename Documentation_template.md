# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

A blocking + classifier pipeline for cross-source business entity resolution.
Candidates are generated per Source-1 record via country-bucketed, character
n-gram TF-IDF cosine similarity (with a token-overlap fallback block), then
scored by a gradient-boosted classifier trained on pairwise string- and
attribute-similarity features. The decision threshold is tuned directly
against the leaderboard's macro F_0.5 metric rather than a generic accuracy
proxy, which matters because F_0.5 punishes false merges twice as hard as
misses.

*[Fill in your final validation F_0.5 score and any team-specific tuning
once you have run the pipeline against the real dataset — see Section 5.]*

---

## 2. Methodology

### 2.1 Problem Analysis

EDA-relevant noise patterns the pipeline is built to absorb (see the problem
statement's "Noise Patterns to Expect"):

- **Name noise:** legal-suffix inconsistency (Corp/Corporation, Pvt/Private,
  Ltd/Limited), `&` vs "and", DBA/trade names, word-order transpositions,
  typos.
- **Address noise:** street-type abbreviations (Rd/Road, St/Street),
  transliteration variants, missing components (no PIN/ZIP, no state),
  landmark references ("Near SBI ATM"), municipal numbering and component
  reordering.
- **Open-set `country` field:** training only contains `US`/`India`; the
  test set adds `France`. The pipeline never hard-codes or one-hot-encodes
  a fixed country list — `country` is used only as an equality predicate
  (for blocking and as a feature), which works identically for any string
  value, seen in training or not.

*[Add any dataset-specific observations from your own EDA once you have
the real files — e.g. field-length distributions, missingness rates,
class balance of matches vs singletons.]*

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (supervised pairwise matching)

**Core Innovation:** The threshold on the classifier's match probability is
selected by directly maximizing macro-averaged F_0.5 on a held-out
validation split (grid search over the threshold, not a fixed 0.5 cutoff),
which aligns training-time model selection with the actual scoring metric
and its precision/recall asymmetry.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - Primary: `country` bucket (exact string match) — two records of the
    same real-world business are expected to share a country label, so
    this is a safe, value-agnostic partition (works the same for a country
    absent from training).
  - Within each country bucket: character n-gram (2-4 char, word-boundary
    aware) TF-IDF over `normalized_name + " " + normalized_address`, with
    top-K (K=15) nearest Source-2/3 neighbors per Source-1 record by cosine
    similarity, above a minimum similarity floor (0.10).
  - Fallback: records sharing the first normalized name token (with at
    least one other shared name token) are unioned in even if below the
    TF-IDF similarity floor, to catch short/noisy names the n-gram
    vectorizer under-scores.
  - Cosine similarity is computed via chunked sparse matrix multiplication
    (fixed-size row chunks) rather than a dense pairwise matrix, so memory
    stays bounded regardless of source size.
- **Candidate pairs generated:** *[fill in the count printed by
  `train.py`/`predict.py` on your real data]*
- **How true matches were not lost:** `train.py` reports a **blocking
  recall ceiling** — the fraction of ground-truth matches present in the
  candidate set before the classifier ever sees them. This is the hard
  upper bound on achievable recall; if it's below your target, widen `K`
  or lower `MIN_SIM` in `src/blocking.py` before touching the classifier.

---

## 4. Matching Model

**Features used** (`src/features.py`):
- Name features: TF-IDF cosine (carried over from blocking), token
  Jaccard, Levenshtein ratio, token-sort ratio (order-invariant, via
  `rapidfuzz`), length delta, exact first-token match.
- Address features: token Jaccard, Levenshtein ratio, length delta,
  exact postal/PIN/ZIP code match (regex-extracted 4-6 digit token).
- Other: exact `country` match.

**Model type:** `sklearn.ensemble.HistGradientBoostingClassifier`, trained
from scratch on the engineered features only (BSD-licensed library, not a
pretrained model — well within the MIT/Apache-2.0, ≤8B-parameter
constraint).

**Threshold selection method:** grid search (0.05-0.95, step 0.02) over the
classifier's positive-class probability, evaluated by macro F_0.5 on a
group-wise train/validation split (split by Source-1 entity so no entity's
pairs appear on both sides).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** *[fill in the validation score printed by
  `train.py` after running on the real dataset]*
- **Common false positives (wrong merges):** *[fill in after inspecting
  validation errors — e.g. distinct businesses at the same address/plaza,
  common generic names]*
- **Common false negatives (missed matches):** *[fill in — e.g. heavy
  transliteration, landmark-only addresses with no shared tokens]*

---

## 6. Conclusion

The pipeline combines value-agnostic, country-bucketed blocking with a
supervised classifier whose decision threshold is tuned against the actual
scoring metric, giving a precision-heavy operating point appropriate for
F_0.5. *[Summarize your final results and any lessons learned once run
against the real dataset.]*

---

## Appendix

### A. Code Artefacts

Runnable pipeline ships under `code/EntityMatch/`:

```
code/EntityMatch/
├── README.md              # exact run instructions
├── requirements.txt       # pinned dependencies
└── src/
    ├── normalize.py        # name/address cleanup, postal-code extraction
    ├── blocking.py          # candidate generation (blocking)
    ├── features.py          # pairwise feature engineering
    ├── model.py             # classifier + macro-F0.5 threshold tuning
    ├── data_io.py           # TSV loading / output writers
    ├── train.py             # entry point: train + tune threshold
    └── predict.py           # entry point: generate output/*.tsv
```

Entry points (run from `student_resource/`, see `README.md` for details):

```bash
python3 code/EntityMatch/src/train.py
python3 code/EntityMatch/src/predict.py
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### B. Additional Results

*[Add charts/tables here — e.g. F_0.5 vs threshold curve, precision/recall
by country, candidate-set size distribution.]*

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
