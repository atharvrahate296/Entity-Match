"""Score a matching_results.tsv-style predictions file against ground truth.

Computes the competition's own metric -- macro-averaged F_0.5, scored per
Source-1 entity then averaged across all entities, singletons included --
plus plain precision/recall so you can see WHY the F_0.5 moved (more false
merges vs. more missed matches) when you tweak the pipeline.

Since the real test set has no ground truth (per the Problem Statement),
this is meant to run on a held-out slice of train_ground_truth.tsv: e.g.
run predict.py-style scoring on your own validation S1 ids, write the
result in matching_results.tsv format, then compare it here against the
matching rows of train_ground_truth.tsv.

Usage:
    python code/src/score_predictions.py \\
        --predictions output/matching_results.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv

Both files must be tab-separated with a source1_entity_id column and a
comma-separated id-list column (matched_entity_ids by default -- pass
--pred-list-col candidate_entity_ids to score candidate_pairs.tsv against
ground truth instead, e.g. to check your blocking recall ceiling on a
validation slice).
"""

import argparse
import csv


def load_id_list(path, id_col, list_col):
    m = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        idx_id = header.index(id_col)
        idx_list = header.index(list_col)
        for row in reader:
            if not row:
                continue
            s1 = row[idx_id]
            ids_str = row[idx_list] if len(row) > idx_list else ""
            ids = set(x for x in ids_str.split(",") if x) if ids_str else set()
            m[s1] = ids
    return m


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="matching_results.tsv (or candidate_pairs.tsv) to score")
    ap.add_argument("--ground-truth", required=True,
                    help="ground truth tsv: source1_entity_id, matched_entity_ids")
    ap.add_argument("--pred-id-col", default="source1_entity_id")
    ap.add_argument("--pred-list-col", default="matched_entity_ids",
                    help="use candidate_entity_ids to score candidate_pairs.tsv "
                         "instead (e.g. to check blocking recall ceiling)")
    ap.add_argument("--only-ids", default=None,
                    help="optional path to a file with one S1 id per line -- "
                         "restrict scoring to just these (e.g. your held-out "
                         "validation ids), ignoring everything else in both files")
    args = ap.parse_args()

    pred = load_id_list(args.predictions, args.pred_id_col, args.pred_list_col)
    truth = load_id_list(args.ground_truth, "source1_entity_id", "matched_entity_ids")

    if args.only_ids:
        with open(args.only_ids, encoding="utf-8") as f:
            keep = set(line.strip() for line in f if line.strip())
        pred = {k: v for k, v in pred.items() if k in keep}
        truth = {k: v for k, v in truth.items() if k in keep}
        all_ids = keep
    else:
        all_ids = set(pred) | set(truth)

    scores = []
    tp_total = fp_total = fn_total = 0
    n_singleton_correct = n_singleton_total = 0
    n_false_merge_on_singleton = 0

    for s1 in all_ids:
        p = pred.get(s1, set())
        t = truth.get(s1, set())
        tp = len(p & t)
        fp = len(p - t)
        fn = len(t - p)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        if not t:
            n_singleton_total += 1
            if not p:
                n_singleton_correct += 1
            else:
                n_false_merge_on_singleton += 1

        if not t and not p:
            scores.append(1.0)
            continue
        if not p:
            scores.append(0.0)
            continue
        precision = tp / len(p) if p else 0.0
        recall = tp / len(t) if t else 0.0
        scores.append(f_beta(precision, recall, beta=0.5))

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0

    print(f"Entities scored:                  {len(all_ids)}")
    print(f"Macro F_0.5 (leaderboard metric):  {macro_f05:.4f}")
    print(f"Micro precision:                  {micro_precision:.4f}")
    print(f"Micro recall:                     {micro_recall:.4f}")
    print(f"Singletons correctly empty:        {n_singleton_correct}/{n_singleton_total}")
    print(f"False merges on true singletons:   {n_false_merge_on_singleton}")


if __name__ == "__main__":
    main()