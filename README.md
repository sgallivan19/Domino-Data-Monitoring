# Domino-Data-Monitoring

A proof of concept for monitoring clinical study data. A Streamlit dashboard shows two kinds of checks on synthetic ADaM data for a fictional Phase 3 study, **XYZ-301**:

| Tab | Type | What it does |
|---|---|---|
| **QTL Monitoring** | Deterministic | Tracks the % of randomized (ITT) subjects who discontinued treatment due to an adverse event (`ADSL.DCTREAS = "ADVERSE EVENT"`). The rate is compared against a QTL (default 15%) and a secondary limit (default 10%), both editable in the sidebar. The tab shows the cumulative monthly trend and breakdowns by arm and by site. |
| **AI Data Review** | Non-deterministic (LLM) | Reviews "Reason for study discontinuation" (`DCSREAS`) and its "If other, specify" free text (`DCSREASP`). It flags text that maps to an existing term, conflicts with other data, is uninformative, or contains PII. Each flag shows what is out of place, why, and the full ADSL record plus the subject's ADAE rows. |

## Setup (Python 3.13)

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python data/generate_data.py          # optional: regenerate synthetic data (seeded, reproducible)
export ANTHROPIC_API_KEY=...          # needed only to run the AI review
python -m monitoring.ai_review        # runs the LLM over the candidate records and writes data/ai_flags.json
streamlit run app.py
```

The dashboard reads `data/ai_flags.json`. The sidebar's **Re-run AI review** button refreshes it when credentials are available. On Domino, `app.sh` launches the app on port 8888.

## Data (`data/`)

| File | Contents |
|---|---|
| `adsl.csv` | Subject-level data: 600 subjects, 15 sites, 3 arms (Placebo, XYZ-123 50 mg, XYZ-123 100 mg). |
| `adae.csv` | Adverse events (MedDRA-like PT/SOC, severity, action taken). |
| `ai_seeded_anomalies.csv` | Answer key for the planted "Other, specify" cases (20 problematic, 22 legitimate). **Reference only.** The app and the AI review never read it. |
| `ai_flags.json` | Cached AI review output. |

## Code

- `monitoring/qtl.py`: QTL calculations (overall, by group, cumulative trend, status).
- `monitoring/ai_review.py`: LLM review (`claude-opus-5-5`, structured output via Pydantic, server-side refusal fallback enabled) and caching.
- `app.py`: Streamlit UI.
- `tests/test_qtl.py`: unit tests (`pytest`).
