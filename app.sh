#!/usr/bin/env bash
# Domino App launcher
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${MONITORING_DATA_DIR:-${DOMINO_DATASETS_DIR:-/mnt/data}/Clinical-Data-Monitoring}"
if [ ! -f "$DATA_DIR/adsl.csv" ] || [ ! -f "$DATA_DIR/adae.csv" ]; then
  echo "ERROR: adsl.csv/adae.csv not found in $DATA_DIR. Is the Clinical-Data-Monitoring dataset mounted?" >&2
  exit 1
fi

pip install --user --quiet --disable-pip-version-check -r requirements.txt

exec streamlit run app.py \
  --server.port 8888 --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false
