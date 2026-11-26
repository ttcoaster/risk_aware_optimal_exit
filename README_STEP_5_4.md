# STEP 5-4: Full validation pilot evaluation
Copy files into project root. No existing files replaced. Run:

    python -m unittest discover -s tests -p test_validation_report.py -v
    python -m unittest discover -s tests -p "test_*.py" -v
    python -m scripts.evaluate_validation --model-run smoke01 --run-id validation54

Expected 5264 runs, 376 episodes per policy, 14 policies, 1128 R2-R1 pairs.
Outputs under artifacts/evaluation/validation54. Existing output is not overwritten.
A .partial directory may remain if export fails; use a fresh run ID after resolving errors.
Market groups use realized B0 return sign, for retrospective description only.
No training, test evaluation, significance testing or model selection. No-op rate is pooled across sell requests; no requests yields blank. Missing exit milestones are not treated as zero. Feature hashes describe current files, not proof of training-time identity. Single seed, overlapping episodes, adjusted prices and zero cash interest remain limitations.
