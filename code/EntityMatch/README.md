# Business Entity Resolution — Pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` end to end:
data -> blocking -> feature engineering -> classifier -> thresholded output.

## Setup

Run everything from your `student_resource/` root (the folder that contains
`dataset/`, `utils/`, and this `code/` folder) so the default paths below
line up with the challenge's own directory layout with zero extra flags:

```
student_resource/
├── dataset/{train,test}/...
├── utils/validate_submission.py
├── Documentation_template.md
└── code/EntityMatch/   <- this package
```

```bash
cd student_resource
uv venv env   # Use uv package manager to create virtual environment
.\env\Scripts\activate
uv pip install -r code/EntityMatch/requirements.txt
```

Python 3.9+ recommended (developed/tested on 3.12).

## 1. Train the matching model

From `student_resource/`:

```bash
python code/EntityMatch/src/train.py
```

Equivalent to (defaults shown explicitly):

```bash
python code/EntityMatch/src/train.py \
    --train-dir dataset/train \
    --model-out code/EntityMatch/models/model.joblib
```

This:
- loads `train_source1/2/3.tsv` and `train_ground_truth.tsv`,
- runs blocking to generate candidate pairs and reports the **blocking
  recall ceiling** (fraction of true matches present in the candidate set —
  the upper bound on achievable recall),
- does a group-wise train/validation split by Source-1 entity (no entity
  leaks across the split),
- trains a `HistGradientBoostingClassifier` on the engineered pairwise
  features,
- searches a probability threshold that maximizes macro-averaged F_0.5 on
  the validation split, and
- saves `{classifier, threshold}` to `models/model.joblib`.

## 2. Generate predictions on the test set

From `student_resource/`:

```bash
python code/EntityMatch/src/predict.py
```

Equivalent to (defaults shown explicitly):

```bash
python code/EntityMatch/src/predict.py \
    --test-dir dataset/test \
    --model code/EntityMatch/models/model.joblib \
    --output-dir output
```

Writes `output/candidate_pairs.tsv` (the exact candidate set the model
scored) and `output/matching_results.tsv` (final matches after
thresholding — the file scored on the leaderboard). Every Source-1 entity in
`test_source1.tsv` gets exactly one row in both files; entities with no
candidates/matches get an empty `matched_entity_ids` / `candidate_entity_ids`.

## 3. Validate before submitting

From `student_resource/`, using the challenge's own validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Add `--check-ids` for the stricter (heavier) check that every matched/
candidate ID actually exists in `test_source2.tsv` / `test_source3.tsv`.
A `PASS` here means the files are safe to upload; copy
`output/matching_results.tsv` and `output/candidate_pairs.tsv`, plus this
`code/` folder and the filled-in `Documentation_template.md`, into your
`<team_name>_submission.zip`.

## Pipeline design

- **`src/normalize.py`** — rule-based cleanup of names (legal-suffix
  collapsing: Corp/Corporation, Pvt/Private, Ltd/Limited, `&`/and, ...) and
  addresses (street-type abbreviations, postal/PIN-code extraction). No
  external lookups anywhere — everything is deterministic string
  processing on the provided fields, per the fair-play rules.
- **`src/blocking.py`** — candidate generation. Buckets records by the
  `country` field (treated as an open string label — works identically for
  a country never seen in training, e.g. France at test time), then within
  each bucket uses a character n-gram TF-IDF over `name + address` and
  chunked sparse cosine similarity to pull each Source-1 record's top-K
  nearest Source-2/3 records. A cheap shared-first-token fallback block is
  unioned in to catch cases the n-gram vectorizer under-scores. Chunked
  matrix multiplication keeps memory bounded at large scale.
- **`src/features.py`** — pairwise features: TF-IDF cosine (from blocking),
  token Jaccard, Levenshtein ratio and token-sort ratio (via `rapidfuzz`)
  on both name and address, length deltas, postal-code exact match, and
  country match.
- **`src/model.py`** — `HistGradientBoostingClassifier` (scikit-learn,
  BSD-licensed, trained from scratch — trivially within the "MIT/Apache
  2.0, <=8B parameters" constraint since it isn't a pretrained model at
  all) plus a threshold search that directly optimizes macro F_0.5, the
  actual leaderboard metric.
- **`src/train.py`** / **`src/predict.py`** — CLI entry points described
  above.
- **`src/data_io.py`** — TSV loading and the two output writers.

## Notes on the precision-heavy metric

F_0.5 weights precision 2x over recall, and singletons scored 0 vs 1 on any
false merge, so the threshold search in `train.py` optimizes macro F_0.5
directly rather than a generic accuracy/AUC proxy — this tends to push the
decision threshold higher than a naive 0.5 cutoff, trading some recall on
ambiguous pairs for fewer false merges.
