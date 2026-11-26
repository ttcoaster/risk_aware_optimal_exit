# STEP 5-5: Offline behavior diagnostics
Copy files into project root. No model training or environment changes.

Run `python -m unittest discover -s tests -p test_behavior_diagnostics.py -v`, then full regression, then `python -m scripts.diagnose_validation --source-run validation54 --run-id diagnostics55`.

Eight CSV reports plus run.json are saved under artifacts/evaluation/diagnostics55. Source hashes record the exact inputs. Existing output directories are not overwritten. Requires pandas and numpy. Market groups use realized B0 return only for retrospective analysis. Exit timing means are conditional on reaching the threshold; exit curves use all episodes as denominator. Action and holding tables include observed combinations only. No-op is split into cancelled requests and tradable zero-share requests. This is a trace audit, not independent price-data or model replay verification. No statistical inference or policy changes.
