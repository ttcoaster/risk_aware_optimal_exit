# STEP 6-4 Pilot
Requires STEP 6-3 RNG fix and existing pilot61 plan. No replacement of existing files.
Run `python -m scripts.run_pilot --plan-id pilot61 --run-id pilot64` then
`python -m scripts.evaluate_pilot --training-run pilot64 --run-id pilotval64`.
All six jobs train continuously from scratch for 20480 transitions each. Checkpoints at 2048,10240,20480. Evaluation: 18 checkpoints and 8 fresh benchmarks, 376 validation episodes each. No test performance or automatic selection. CEQ uses existing metrics defaults. Mean noop rate is episode-weighted. DQN exploration schedule depends on total budget, so 2048 pilot checkpoint is not equivalent to 2048 integration run. Partial outputs must be preserved and new IDs used on failure. Checkpoint files are inference snapshots, not complete resumable training states.
