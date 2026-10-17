# Domino-Data-Monitoring

A proof of concept for monitoring clinical study data. A Streamlit dashboard shows two kinds of checks on synthetic ADaM data for a fictional Phase 3 study, **XYZ-301**:

| Tab | Type | What it does |
|---|---|---|
| **QTL Monitoring** | Deterministic | Tracks the % of randomized (ITT) subjects who discontinued treatment due to an adverse event (`ADSL.DCTREAS = "ADVERSE EVENT"`). The rate is compared against a QTL (default 15%) and a secondary limit (default 10%), both editable in the sidebar. The tab shows the cumulative monthly trend and breakdowns by arm and by site. |
| **AI Data Review** | Non-deterministic (LLM) | Reviews "Reason for study discontinuation" (`DCSREAS`) and its "If other, specify" free text (`DCSREASP`). It flags text that maps to an existing term, conflicts with other data, is uninformative, or contains PII. Each flag shows what is out of place, why, and the full ADSL record plus the subject's ADAE rows. |

## Setup (Python 3.12+)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

The data lives in the **Clinical-Data-Monitoring** Domino dataset, mounted at `/mnt/data/Clinical-Data-Monitoring/` (from `$DOMINO_DATASETS_DIR`). To read from somewhere else, such as a snapshot, set `MONITORING_DATA_DIR` to that folder.

```bash
python -m monitoring.ai_review        # AI review: needs ANTHROPIC_API_KEY; writes ai_flags.json to the dataset
streamlit run app.py                  # dashboard (read-only)
```

The dashboard only reads the AI review output. The review itself runs as a Domino Job, so each run is logged and can be scheduled.

## Deploy on Domino

1. **AI review Job:** in **Jobs → Run**, use the command `bash run_ai_review.sh`. Optionally, add a **Scheduled Job** with the same command, sequential run mode, and a Quartz cron such as `0 0 2 * * ?` (daily at 02:00). The Job needs `ANTHROPIC_API_KEY` as a Domino environment variable. It fails without overwriting the previous results if every record errors.
2. **App:** in **Deployments → Apps → Publish**, use the launch file `app.sh`. It checks that the dataset is mounted, installs `requirements.txt`, and serves Streamlit on port 8888. The App doesn't need the API key.

## Data (Domino dataset)

| File | Contents |
|---|---|
| `adsl.csv` | Subject-level data: 600 subjects, 15 sites, 3 arms (Placebo, XYZ-123 50 mg, XYZ-123 100 mg). |
| `adae.csv` | Adverse events (MedDRA-like PT/SOC, severity, action taken). |
| `ai_flags.json` | AI review output, written by the AI review Job. |

## Code

- `monitoring/qtl.py`: QTL calculations (overall, by group, cumulative trend, status).
- `monitoring/ai_review.py`: LLM review (`claude-opus-5-5`, structured output via Pydantic, server-side refusal fallback enabled). Also the Job entrypoint.
- `app.py`: Streamlit UI.
- `app.sh` / `run_ai_review.sh`: Domino App launcher and AI review Job command.
- `tests/test_qtl.py`: unit tests (`pytest`).
