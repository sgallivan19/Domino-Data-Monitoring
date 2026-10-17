#!/usr/bin/env bash
# Domino Job: run the AI review and write ai_flags.json to the dataset
set -euo pipefail
cd "$(dirname "$0")"
pip install --user --quiet --disable-pip-version-check -r requirements.txt
exec python -m monitoring.ai_review
