"""AI review of "Reason for study discontinuation" free text (DCSREASP) using an LLM.

Runs as a Domino Job (run_ai_review.sh) and writes ai_flags.json to the Domino dataset, which the
dashboard reads. From the repo root:
    python -m monitoring.ai_review
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_DIR = Path(__file__).resolve().parent.parent
DATASET_NAME = "Clinical-Data-Monitoring"
# Domino dataset mount (Git-based projects: /mnt/data/<name>). MONITORING_DATA_DIR overrides, e.g. for a snapshot.
DATA_DIR = Path(
    os.environ.get("MONITORING_DATA_DIR") or Path(os.environ.get("DOMINO_DATASETS_DIR", "/mnt/data")) / DATASET_NAME
)
CACHE_PATH = DATA_DIR / "ai_flags.json"
MODEL = "claude-opus-5-5"
MAX_WORKERS = 8

# Load ANTHROPIC_API_KEY from the repo-root .env (existing environment variables take precedence).
load_dotenv(REPO_DIR / ".env")

DCSREAS_TERMS = [
    "ADVERSE EVENT", "LACK OF EFFICACY", "LOST TO FOLLOW-UP", "WITHDRAWAL BY SUBJECT",
    "PHYSICIAN DECISION", "PROTOCOL DEVIATION", "DEATH", "OTHER",
]

Category = Literal[
    "NONE",
    "MISCLASSIFIED_REASON",
    "CONFLICTS_WITH_DATA",
    "UNINTELLIGIBLE_OR_UNINFORMATIVE",
    "PII_OR_INAPPROPRIATE",
    "OTHER",
]


class ReviewResult(BaseModel):
    flagged: bool = Field(description="True if the record needs data manager follow-up.")
    severity: Literal["High", "Medium", "Low", "None"] = Field(
        description="High: likely wrong reason or factual conflict; Medium: missing/unclear information "
        "or PII; Low: minor wording issue; None: not flagged."
    )
    category: Category = Field(description="Primary issue category, NONE if not flagged.")
    out_of_place: str = Field(description="What specifically is out of place (one sentence). Empty if not flagged.")
    rationale: str = Field(description="Why, citing the specific fields and values used. 1-3 sentences.")
    suggested_reason: str | None = Field(
        description="Controlled term the record should probably use instead, if any."
    )
    related_fields: list[str] = Field(description="ADSL/ADAE variable names that support the finding.")


SYSTEM_PROMPT = f"""You are an experienced clinical data manager reviewing ADaM subject-level data \
(ADSL, with the subject's ADAE records) for a randomized Phase 3 trial.

Your task: review the study discontinuation reason (DCSREAS) and its free-text "If other, specify" field \
(DCSREASP) for ONE subject and decide whether the record needs a data query.

DCSREAS controlled terms: {", ".join(DCSREAS_TERMS)}.

Flag the record when any of these apply (pick the most important as the category):
- MISCLASSIFIED_REASON: the free text describes a situation that an existing controlled term already covers \
(e.g. an adverse event, consent withdrawal, inability to contact the subject, lack of efficacy, investigator \
decision, eligibility/protocol violation, death), so OTHER should not have been used.
- CONFLICTS_WITH_DATA: the free text contradicts other variables (e.g. dates, SEX, death flags DTHFL/DTHDT/\
LSTALVDT, EOSSTT, TRTSDT/SAFFL, or the subject's ADAE records), or specify text is present but contradicts a \
non-OTHER DCSREAS.
- UNINTELLIGIBLE_OR_UNINFORMATIVE: the text is blank-equivalent, gibberish, a placeholder, or does not state a \
reason (e.g. "N/A", "see source", "other").
- PII_OR_INAPPROPRIATE: the text contains personal identifiers (names, phone numbers, addresses, emails) or \
otherwise inappropriate content.

Do NOT flag a record when the free text gives a clear, plausible reason that no controlled term covers \
(e.g. relocation with no nearby site, site closure, incarceration) and nothing in the data contradicts it. \
Do not flag on wording or grammar alone. Note that specify text that is consistent with a non-OTHER DCSREAS \
is acceptable.

Judge only from the data provided; do not assume facts that are not in the record. When not flagged, set \
severity to "None", category to "NONE", out_of_place to "", and give a one-sentence rationale."""


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [name for name in ("adsl.csv", "adae.csv") if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not found in {DATA_DIR}. Check that the {DATASET_NAME} dataset is mounted, "
            "or set MONITORING_DATA_DIR to the folder that holds the data."
        )
    read = lambda name: pd.read_csv(DATA_DIR / name, dtype=str, keep_default_na=False)  # noqa: E731
    return read("adsl.csv"), read("adae.csv")


def records_to_review(adsl: pd.DataFrame) -> pd.DataFrame:
    """Deterministic pre-filter: reason is OTHER, or specify text is present at all."""
    mask = (adsl["DCSREAS"] == "OTHER") | (adsl["DCSREASP"].str.strip() != "")
    return adsl[mask]


def build_prompt(subject: dict, aes: list[dict]) -> str:
    return (
        "Review this subject's study discontinuation record.\n\n"
        f"ADSL record:\n{json.dumps(subject, indent=2)}\n\n"
        f"ADAE records ({len(aes)}):\n{json.dumps(aes, indent=2) if aes else 'None'}"
    )


def has_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def make_client() -> anthropic.Anthropic:
    # API keys not scoped to a workspace must name one via the anthropic-workspace-id header.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    return anthropic.Anthropic(max_retries=4, default_headers=headers)


def review_one(client: anthropic.Anthropic, subject: dict, aes: list[dict]) -> dict:
    base = {"USUBJID": subject["USUBJID"]}
    try:
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            # Server-side fallback: if a request is declined by a safety classifier, the API re-runs it
            # on an appropriate fallback model within the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(subject, aes)}],
            output_format=ReviewResult,
        )
    except anthropic.RateLimitError as e:
        return {**base, "status": "error", "error": f"Rate limited after retries: {e}"}
    except anthropic.APIStatusError as e:
        return {**base, "status": "error", "error": f"API error {e.status_code}: {e.message}"}
    except anthropic.APIConnectionError as e:
        return {**base, "status": "error", "error": f"Connection error: {e}"}

    if response.stop_reason == "refusal" or response.parsed_output is None:
        return {**base, "status": "error", "error": f"No structured result (stop_reason={response.stop_reason})"}
    return {**base, "status": "ok", **response.parsed_output.model_dump()}


def run_review(max_workers: int = MAX_WORKERS) -> dict:
    """Review all candidate records and write CACHE_PATH, unless every record errored (keeps the last good run)."""
    adsl, adae = load_data()
    targets = records_to_review(adsl)
    ae_by_subj = {k: g.drop(columns=["STUDYID"]).to_dict("records") for k, g in adae.groupby("USUBJID")}
    client = make_client()

    jobs = [(row, ae_by_subj.get(row["USUBJID"], [])) for row in targets.to_dict("records")]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda job: review_one(client, *job), jobs))

    payload = {
        "model": MODEL,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_reviewed": len(results),
        "n_flagged": sum(r.get("flagged", False) for r in results),
        "n_errors": sum(r["status"] == "error" for r in results),
        "results": results,
    }
    if results and payload["n_errors"] == len(results):
        return payload
    # Write then rename so the dashboard never reads a half-written file.
    tmp = CACHE_PATH.with_name(CACHE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, CACHE_PATH)
    return payload


def load_cache() -> dict | None:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else None


if __name__ == "__main__":
    if not has_credentials():
        sys.exit("ANTHROPIC_API_KEY is not set. Add it as a Domino environment variable and re-run.")
    print(f"Data: {DATA_DIR}", flush=True)
    out = run_review()
    if out["n_reviewed"] and out["n_errors"] == out["n_reviewed"]:
        print(f"All {out['n_reviewed']} records failed; kept the previous {CACHE_PATH}. First error: "
              f"{out['results'][0]['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"Reviewed {out['n_reviewed']} records: {out['n_flagged']} flagged, {out['n_errors']} errors "
          f"-> {CACHE_PATH}")
