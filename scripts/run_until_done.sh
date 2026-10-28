#!/bin/bash
# Runs a batch to completion, then scores it once. run_batch.py --until-done calls run_batch every
# run.retry_interval_s (10 minutes) until no planned run is missing; a run that used all run.max_attempts gets an
# empty submission, so it no longer counts as missing.
# Usage, with the project venv active:  scripts/run_until_done.sh <batch id> [--config <file>]
set -euo pipefail
cd "$(dirname "$0")/.."
batch=${1:?usage: scripts/run_until_done.sh <batch id> [--config <file>]}
shift
python scripts/run_batch.py --batch "$batch" --until-done "$@"
python scripts/score_batch.py --batch "$batch" "$@"
