import pandas as pd
import pytest

from monitoring import qtl


@pytest.fixture
def adsl() -> pd.DataFrame:
    rows = [
        # USUBJID, SITEID, ARM, ITTFL, RANDDT, TRTEDT, DCTREAS
        ("S1", "101", "A", "Y", "2024-01-10", "2024-03-05", "ADVERSE EVENT"),
        ("S2", "101", "A", "Y", "2024-01-20", "", ""),
        ("S3", "102", "B", "Y", "2024-02-10", "2024-04-15", "ADVERSE EVENT"),
        ("S4", "102", "B", "Y", "2024-02-15", "2024-03-01", "LOST TO FOLLOW-UP"),
        ("S5", "102", "B", "N", "2024-02-20", "2024-03-01", "ADVERSE EVENT"),  # not ITT
    ]
    return pd.DataFrame(rows, columns=["USUBJID", "SITEID", "ARM", "ITTFL", "RANDDT", "TRTEDT", "DCTREAS"])


def test_overall_rate_uses_itt_and_ae_only(adsl):
    r = qtl.overall_rate(adsl)
    assert r == {"n": 4, "events": 2, "rate": 0.5}


@pytest.mark.parametrize(
    "rate,expected",
    [(0.05, qtl.OK), (0.10, qtl.OK), (0.12, qtl.SECONDARY_EXCEEDED), (0.15, qtl.SECONDARY_EXCEEDED),
     (0.16, qtl.QTL_BREACHED)],
)
def test_status_thresholds(rate, expected):
    assert qtl.status(rate, qtl=0.15, secondary=0.10) == expected


def test_rate_by_group(adsl, monkeypatch):
    monkeypatch.setattr(qtl, "MIN_GROUP_N", 1)
    out = qtl.rate_by(adsl, "SITEID", qtl=0.15, secondary=0.10).set_index("SITEID")
    assert out.loc["101", "n"] == 2 and out.loc["101", "events"] == 1
    assert out.loc["102", "n"] == 2 and out.loc["102", "events"] == 1
    assert out.loc["102", "status"] == qtl.QTL_BREACHED


def test_rate_by_small_n_has_no_status(adsl):
    out = qtl.rate_by(adsl, "ARM", qtl=0.15, secondary=0.10)
    assert set(out["status"]) == {qtl.SMALL_N}


def test_cumulative_trend(adsl):
    trend = qtl.cumulative_trend(adsl)
    assert trend["n"].is_monotonic_increasing
    assert trend["events"].is_monotonic_increasing
    last = trend.iloc[-1]
    assert (last["n"], last["events"]) == (4, 2)
    jan = trend[trend["date"] == "2024-01-31"].iloc[0]
    assert (jan["n"], jan["events"]) == (2, 0)
