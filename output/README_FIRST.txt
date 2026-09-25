This folder is where `matching_results.tsv` and `candidate_pairs.tsv` belong
in your final submission zip.

They are NOT included here because this package was built without access to
your actual dataset/train and dataset/test files — only the problem
statement, README, documentation template, and validator script were
available when this pipeline was built.

To generate the real files, from your `student_resource/` root:

    python code/src/train.py
    python code/src/predict.py
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Once validate_submission.py prints PASS, delete this note and zip this
output/ folder (containing the two real .tsv files) together with code/
and Documentation_template.md.
