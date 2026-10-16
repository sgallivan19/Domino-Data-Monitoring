"""Deterministic QTL: % of randomized (ITT) subjects who discontinued treatment due to an AE."""

from __future__ import annotations

import pandas as pd

AE_REASON = "ADVERSE EVENT"
DEFAULT_QTL = 0.15
DEFAULT_SECONDARY = 0.10
MIN_GROUP_N = 20  # below this, a group rate is shown but not assigned a status

OK = "OK"
SECONDARY_EXCEEDED = "SECONDARY LIMIT EXCEEDED"
QTL_BREACHED = "QTL BREACHED"
SMALL_N = "SMALL N"


def _itt(adsl: pd.DataFrame) -> pd.DataFrame:
    return adsl[adsl["ITTFL"] == "Y"]


def _is_ae_disc(adsl: pd.DataFrame) -> pd.Series:
    return adsl["DCTREAS"] == AE_REASON


def status(rate: float, qtl: float, secondary: float) -> str:
    if rate > qtl:
        return QTL_BREACHED
    if rate > secondary:
        return SECONDARY_EXCEEDED
    return OK


def overall_rate(adsl: pd.DataFrame) -> dict:
    itt = _itt(adsl)
    n, events = len(itt), int(_is_ae_disc(itt).sum())
    return {"n": n, "events": events, "rate": events / n if n else 0.0}


def rate_by(adsl: pd.DataFrame, col: str, qtl: float, secondary: float) -> pd.DataFrame:
    itt = _itt(adsl).assign(_event=lambda d: _is_ae_disc(d))
    out = itt.groupby(col).agg(n=("_event", "size"), events=("_event", "sum")).reset_index()
    out["rate"] = out["events"] / out["n"]
    out["status"] = [
        status(r, qtl, secondary) if n >= MIN_GROUP_N else SMALL_N
        for r, n in zip(out["rate"], out["n"])
    ]
    return out


def cumulative_trend(adsl: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Month-end snapshots: subjects randomized by that date vs AE discontinuations by that date."""
    itt = _itt(adsl)
    randdt = pd.to_datetime(itt["RANDDT"])
    disc_dt = pd.to_datetime(itt.loc[_is_ae_disc(itt), "TRTEDT"])
    end = max(randdt.max(), disc_dt.max() if len(disc_dt) else randdt.max())
    dates = pd.date_range(randdt.min(), end, freq=freq)
    if len(dates) == 0 or dates[-1] < end:
        dates = dates.append(pd.DatetimeIndex([end]))

    rows = []
    for d in dates:
        n = int((randdt <= d).sum())
        events = int((disc_dt <= d).sum())
        rows.append({"date": d, "n": n, "events": events, "rate": events / n if n else 0.0})
    return pd.DataFrame(rows)


def contributing_subjects(adsl: pd.DataFrame, adae: pd.DataFrame) -> pd.DataFrame:
    """AE-discontinued subjects with the AE(s) that led to drug withdrawal."""
    subj = _itt(adsl)[_is_ae_disc(_itt(adsl))]
    withdrawn = (
        adae[adae["AEACN"] == "DRUG WITHDRAWN"]
        .groupby("USUBJID")["AEDECOD"].agg("; ".join)
        .rename("AE LEADING TO WITHDRAWAL")
    )
    return (
        subj[["USUBJID", "SITEID", "COUNTRY", "ARM", "TRTSDT", "TRTEDT", "DCTREAS"]]
        .merge(withdrawn, left_on="USUBJID", right_index=True, how="left")
        .fillna({"AE LEADING TO WITHDRAWAL": ""})
        .sort_values("TRTEDT")
        .reset_index(drop=True)
    )
