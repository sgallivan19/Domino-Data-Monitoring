"""Study data monitoring POC: deterministic QTL monitoring + AI review of free-text discontinuation reasons."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitoring import ai_review, qtl

st.set_page_config(page_title="XYZ-301 Data Monitoring", page_icon=":material/monitoring:", layout="wide")

SERIES = "#2a78d6"
STATUS_COLORS = {
    qtl.OK: "#0ca30c",
    qtl.SECONDARY_EXCEEDED: "#fab219",
    qtl.QTL_BREACHED: "#d03b3b",
    qtl.SMALL_N: "#a3a29c",
}
STATUS_BADGE = {
    qtl.OK: ("green", ":material/check_circle:"),
    qtl.SECONDARY_EXCEEDED: ("orange", ":material/warning:"),
    qtl.QTL_BREACHED: ("red", ":material/error:"),
}
SEVERITY_ORDER = ["High", "Medium", "Low"]


@st.cache_data(ttl="10m")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return ai_review.load_data()


try:
    adsl, adae = load_data()
except FileNotFoundError as e:
    st.error(str(e), icon=":material/error:")
    st.stop()
cache = ai_review.load_cache()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Study XYZ-301")
    st.caption(
        "Phase 3, randomized, double-blind · 3 arms · "
        f"{adsl['SITEID'].nunique()} sites · {adsl['COUNTRY'].nunique()} countries"
    )
    st.caption(f"Data cutoff: {max(adsl['LSTALVDT'])}")

    st.subheader("QTL parameters")
    st.caption("Parameter: % of randomized subjects discontinuing treatment due to an adverse event")
    qtl_limit = st.number_input("QTL (%)", 0.0, 100.0, qtl.DEFAULT_QTL * 100, 0.5) / 100
    secondary = st.number_input("Secondary limit (%)", 0.0, 100.0, qtl.DEFAULT_SECONDARY * 100, 0.5) / 100
    if secondary > qtl_limit:
        st.warning("Secondary limit is above the QTL.")

    st.subheader("AI review")
    if cache:
        st.caption(f"Last run: {cache['run_at']} · Model: `{cache['model']}`")
    else:
        st.caption("No AI review results yet.")
    st.caption("Refreshed by the **AI review** Domino Job (`run_ai_review.sh`).")

tab_qtl, tab_ai = st.tabs([":material/monitoring: QTL Monitoring", ":material/smart_toy: AI Data Review"])


# ---------------------------------------------------------------------------
# Tab 1: QTL monitoring
# ---------------------------------------------------------------------------
def threshold_lines(fig: go.Figure) -> None:
    fig.add_hline(y=qtl_limit * 100, line=dict(color=STATUS_COLORS[qtl.QTL_BREACHED], width=2, dash="dash"),
                  annotation_text=f"QTL {qtl_limit:.1%}", annotation_position="top left")
    fig.add_hline(y=secondary * 100, line=dict(color=STATUS_COLORS[qtl.SECONDARY_EXCEEDED], width=2, dash="dot"),
                  annotation_text=f"Secondary limit {secondary:.1%}", annotation_position="bottom left")


def group_bar(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    fig = go.Figure()
    for st_name, color in STATUS_COLORS.items():
        sub = df[df["status"] == st_name]
        if sub.empty:
            continue
        fig.add_bar(
            x=sub[col], y=sub["rate"] * 100, name=st_name, marker=dict(color=color, cornerradius=4),
            customdata=sub[["events", "n"]],
            hovertemplate="%{x}<br>%{y:.1f}% (%{customdata[0]}/%{customdata[1]})<extra>" + st_name + "</extra>",
        )
    threshold_lines(fig)
    fig.update_layout(
        title=title, yaxis_title="AE discontinuation rate (%)", barmode="overlay", bargap=0.35,
        legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10), height=380,
        xaxis=dict(type="category", categoryorder="array", categoryarray=list(df[col])),
    )
    return fig


with tab_qtl:
    st.subheader("Treatment discontinuation due to adverse events")
    overall = qtl.overall_rate(adsl)
    status = qtl.status(overall["rate"], qtl_limit, secondary)
    color, icon = STATUS_BADGE[status]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Randomized (ITT)", overall["n"])
    c2.metric("AE discontinuations", overall["events"])
    c3.metric("Current rate", f"{overall['rate']:.1%}", f"{(overall['rate'] - qtl_limit) * 100:+.1f} pts vs QTL",
              delta_color="inverse")
    with c4:
        st.markdown("**Status**")
        st.badge(status, color=color, icon=icon)

    trend = qtl.cumulative_trend(adsl)
    fig = go.Figure(go.Scatter(
        x=trend["date"], y=trend["rate"] * 100, mode="lines+markers", name="Cumulative rate",
        line=dict(color=SERIES, width=2), marker=dict(size=8),
        customdata=trend[["events", "n"]],
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}% (%{customdata[0]}/%{customdata[1]})<extra></extra>",
    ))
    threshold_lines(fig)
    fig.update_layout(title="Cumulative AE discontinuation rate (month end)", yaxis_title="Rate (%)",
                      hovermode="x unified", height=380, margin=dict(t=50, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1, 2])
    by_arm = qtl.rate_by(adsl, "ARM", qtl_limit, secondary)
    by_site = qtl.rate_by(adsl, "SITEID", qtl_limit, secondary)
    left.plotly_chart(group_bar(by_arm, "ARM", "By treatment arm"), width="stretch")
    right.plotly_chart(group_bar(by_site, "SITEID", "By site"), width="stretch")
    st.caption(f"Groups with fewer than {qtl.MIN_GROUP_N} subjects are shown as SMALL N and not assigned a status. "
               "Arm-level rates are shown for monitoring only; review in line with blinding requirements.")

    with st.expander("Rate tables"):
        fmt = {"rate": "{:.1%}"}
        t1, t2 = st.columns([1, 2])
        t1.dataframe(by_arm.style.format(fmt), hide_index=True, width="stretch")
        t2.dataframe(by_site.style.format(fmt), hide_index=True, width="stretch")

    with st.expander(f"Contributing subjects ({overall['events']})"):
        st.dataframe(qtl.contributing_subjects(adsl, adae), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 2: AI data review
# ---------------------------------------------------------------------------
def wrap_row_height(df: pd.DataFrame, chars_per_line: int, max_lines: int = 10) -> int | None:
    """Row height that fits the longest text cell when wrapped.

    st.dataframe only wraps text when row_height exceeds 4rem (64px), so return at least 68px, or None
    (default single-line rows) when all text fits on one line.
    """
    longest = max((df[c].astype(str).str.len().max() for c in df.columns), default=0)
    lines = min(-(-longest // chars_per_line), max_lines)
    return None if lines <= 1 else max(lines * 20 + 12, 68)


def highlight_rows(df: pd.DataFrame, fields: list[str]):
    return df.style.apply(
        lambda row: ["background-color: rgba(250, 178, 25, 0.35)" if row["Variable"] in fields else ""] * len(row),
        axis=1,
    )


with tab_ai:
    st.subheader("AI review: study discontinuation reason, \"If other, specify\"")
    st.caption(
        "Records where DCSREAS = OTHER, or the specify text (DCSREASP) is populated, are sent to an LLM with the "
        "full ADSL record and the subject's ADAE records. The LLM flags text that is misclassified, conflicts with "
        "other data, is uninformative, or contains PII."
    )
    st.info("AI flags are review suggestions. A data manager must confirm each one before raising a query.",
            icon=":material/info:")

    if not cache:
        st.warning("No AI review results yet. Run the **AI review** Domino Job (`bash run_ai_review.sh`).")
        st.stop()

    results = pd.DataFrame(cache["results"])
    ok = results[results["status"] == "ok"].astype({"flagged": bool})
    errors = results[results["status"] == "error"]
    flagged = ok[ok["flagged"]]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records reviewed", cache["n_reviewed"])
    c2.metric("Flagged", len(flagged))
    for col, sev in zip((c3, c4, c5), SEVERITY_ORDER):
        col.metric(f"{sev} severity", int((flagged["severity"] == sev).sum()))
    st.caption(f"Model: `{cache['model']}` · Run: {cache['run_at']}")
    if len(errors):
        st.error(f"{len(errors)} record(s) could not be reviewed.", icon=":material/error:")
        with st.expander("Errors"):
            st.dataframe(errors[["USUBJID", "error"]], hide_index=True, width="stretch",
                         row_height=wrap_row_height(errors[["error"]], 100))

    f1, f2 = st.columns(2)
    sev_filter = f1.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
    cats = sorted(flagged["category"].unique())
    cat_filter = f2.multiselect("Category", cats, default=cats)

    table = (
        flagged[flagged["severity"].isin(sev_filter) & flagged["category"].isin(cat_filter)]
        .merge(adsl[["USUBJID", "SITEID", "DCSREAS", "DCSREASP"]], on="USUBJID")
        .assign(_sev=lambda d: d["severity"].map(SEVERITY_ORDER.index))
        .sort_values(["_sev", "USUBJID"])
        .reset_index(drop=True)
    )
    cols = ["USUBJID", "SITEID", "DCSREAS", "DCSREASP", "severity", "category", "out_of_place"]
    st.markdown("**Flagged records**: select a row to see details")
    event = st.dataframe(
        table[cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        row_height=wrap_row_height(table[["DCSREASP", "out_of_place"]], 45),
        column_config={
            "DCSREASP": st.column_config.TextColumn("DCSREASP", width="large"),
            "out_of_place": st.column_config.TextColumn("Out of place", width="large"),
        },
    )

    if event.selection.rows:
        rec = table.iloc[event.selection.rows[0]]
        usubjid = rec["USUBJID"]
        st.divider()
        st.markdown(f"### {usubjid}")
        b1, b2 = st.columns([1, 3])
        b1.badge(f"{rec['severity']} severity",
                 color={"High": "red", "Medium": "orange", "Low": "gray"}[rec["severity"]])
        b2.badge(rec["category"], color="blue")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**What is out of place**")
            st.write(rec["out_of_place"])
            st.markdown("**Why**")
            st.write(rec["rationale"])
            if rec.get("suggested_reason"):
                st.markdown(f"**Suggested DCSREAS:** `{rec['suggested_reason']}`")
            st.markdown("**Related fields:** " + ", ".join(f"`{f}`" for f in rec["related_fields"]))
        with d2:
            st.markdown("**Discontinuation entry as recorded**")
            st.markdown(f"- DCSREAS: `{rec['DCSREAS']}`\n- DCSREASP: \"{rec['DCSREASP']}\"")

        record = adsl[adsl["USUBJID"] == usubjid].iloc[0]
        rec_df = pd.DataFrame({"Variable": record.index, "Value": record.values})
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown("**Full ADSL record** (related fields highlighted)")
            st.dataframe(highlight_rows(rec_df, list(rec["related_fields"])), hide_index=True,
                         width="stretch", height=600, row_height=wrap_row_height(rec_df[["Value"]], 40),
                         column_config={"Value": st.column_config.TextColumn("Value", width="large")})
        with r2:
            subj_ae = adae[adae["USUBJID"] == usubjid]
            st.markdown(f"**ADAE records** ({len(subj_ae)})")
            if subj_ae.empty:
                st.caption("No adverse events recorded.")
            else:
                st.dataframe(subj_ae, hide_index=True, width="stretch",
                             row_height=wrap_row_height(subj_ae[["AEDECOD", "AEBODSYS"]], 25))

    not_flagged = ok[~ok["flagged"]].merge(adsl[["USUBJID", "DCSREAS", "DCSREASP"]], on="USUBJID")
    with st.expander(f"Not flagged ({len(not_flagged)}): AI rationale"):
        st.dataframe(
            not_flagged[["USUBJID", "DCSREAS", "DCSREASP", "rationale"]], hide_index=True, width="stretch",
            row_height=wrap_row_height(not_flagged[["DCSREASP", "rationale"]], 80),
            column_config={
                "DCSREASP": st.column_config.TextColumn("DCSREASP", width="medium"),
                "rationale": st.column_config.TextColumn("Rationale", width="large"),
            },
        )
