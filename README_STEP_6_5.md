# STEP 6-5
Use the existing full61 plan and locally corrected STEP 6-3 RNG-preserving engine. No existing files replaced.

Run `python -m unittest discover -s tests -p test_full_training.py -v`, then full regression.
Run `python -m scripts.run_full_training --plan-id full61 --run-id full65`.
Run `python -m scripts.evaluate_full_training --training-run full65 --run-id fullval65`.
Expected: 30 jobs, 90 checkpoints, 3072000 train steps; 98 policies x 376 = 36848 validation episodes. Seed summary contains 18 groups with sample SD across five training seeds. No test evaluation or selection. Existing output directories rejected. No resume: retain failed runs and use a fresh ID. Evaluation uses existing engine and loads all 90 models in memory; large CSV output. No independent market accounting audit or inference replay performed here. Unit tests cover plan grid only; local end-to-end required.
