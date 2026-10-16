"""Generate synthetic ADaM-style datasets (ADSL, ADAE) for the fictional study XYZ-301.

Outputs (written next to this script):
  adsl.csv                 subject-level analysis dataset
  adae.csv                 adverse events analysis dataset
  ai_seeded_anomalies.csv  answer key of planted "Other, specify" cases (reference only;
                           not read by the app or the AI review)

Run:  python data/generate_data.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
SEED = 42
STUDYID = "XYZ-301"
N_SUBJECTS = 600
FIRST_RAND = date(2024, 1, 15)
LAST_RAND = date(2025, 7, 15)
DATA_CUTOFF = date(2025, 12, 31)
TREATMENT_DAYS = 364  # 52 weeks
FOLLOW_UP_DAYS = 28

SITES = {
    "101": "USA", "102": "USA", "103": "USA", "104": "USA", "105": "USA",
    "106": "CAN", "107": "CAN",
    "108": "DEU", "109": "DEU", "110": "DEU",
    "111": "ESP", "112": "ESP",
    "113": "JPN", "114": "JPN", "115": "JPN",
}
# Sites with an elevated AE-discontinuation signal (the QTL "story").
HOT_SITES = {"108": 2.4, "112": 2.0}

ARMS = [
    # ARMCD, ARM, P(AE discontinuation over 52 wks)
    ("PBO", "Placebo", 0.055),
    ("D50", "XYZ-123 50 mg", 0.095),
    ("D100", "XYZ-123 100 mg", 0.165),
]

# Non-AE treatment discontinuation reasons and their probabilities.
OTHER_DISC = {
    "LACK OF EFFICACY": 0.030,
    "WITHDRAWAL BY SUBJECT": 0.045,
    "LOST TO FOLLOW-UP": 0.020,
    "PHYSICIAN DECISION": 0.015,
    "PROTOCOL DEVIATION": 0.010,
    "DEATH": 0.005,
}

REASON_TERMS = [
    "ADVERSE EVENT", "LACK OF EFFICACY", "LOST TO FOLLOW-UP", "WITHDRAWAL BY SUBJECT",
    "PHYSICIAN DECISION", "PROTOCOL DEVIATION", "DEATH", "OTHER",
]

# (AEDECOD, AEBODSYS)
AE_TERMS = [
    ("Nausea", "Gastrointestinal disorders"),
    ("Diarrhoea", "Gastrointestinal disorders"),
    ("Vomiting", "Gastrointestinal disorders"),
    ("Headache", "Nervous system disorders"),
    ("Dizziness", "Nervous system disorders"),
    ("Fatigue", "General disorders and administration site conditions"),
    ("Nasopharyngitis", "Infections and infestations"),
    ("Upper respiratory tract infection", "Infections and infestations"),
    ("Rash", "Skin and subcutaneous tissue disorders"),
    ("Pruritus", "Skin and subcutaneous tissue disorders"),
    ("Arthralgia", "Musculoskeletal and connective tissue disorders"),
    ("Back pain", "Musculoskeletal and connective tissue disorders"),
    ("Insomnia", "Psychiatric disorders"),
    ("Hypertension", "Vascular disorders"),
    ("Alanine aminotransferase increased", "Investigations"),
    ("Aspartate aminotransferase increased", "Investigations"),
]
AE_DISC_TERMS = [
    ("Alanine aminotransferase increased", "Investigations"),
    ("Rash", "Skin and subcutaneous tissue disorders"),
    ("Nausea", "Gastrointestinal disorders"),
    ("Vomiting", "Gastrointestinal disorders"),
    ("Dizziness", "Nervous system disorders"),
    ("Fatigue", "General disorders and administration site conditions"),
]

LEGIT_TRT_OTHER = [
    "Subject relocated out of state",
    "Sponsor decision to close site",
    "Subject unable to travel to site due to transportation issues",
]


def iso(d: date | None) -> str:
    return d.isoformat() if d else ""


def rand_date(rng: np.random.Generator, start: date, end: date) -> date:
    return start + timedelta(days=int(rng.integers(0, (end - start).days + 1)))


# ---------------------------------------------------------------------------
# Planted "Reason for study discontinuation = OTHER" scenarios
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    text: str
    category: str  # NONE for legitimate entries
    description: str
    dcsreas: str = "OTHER"
    overrides: dict = field(default_factory=dict)  # tag -> handled in apply_scenario


LEGIT = [
    "Subject relocated to another country for work; unable to attend site visits.",
    "Site closed by sponsor decision; no alternate site within reasonable travel distance.",
    "Subject incarcerated; unable to attend study visits.",
    "Subject's primary caregiver passed away; subject no longer able to travel to the site.",
    "Subject moved to a long-term care facility outside the site's catchment area.",
    "Subject began military deployment overseas.",
    "Home destroyed in flooding; subject relocated to temporary housing in another region.",
    "Site closed following investigator retirement; no alternate site available to subject.",
    "Subject's insurance no longer covers transport to site; site unable to arrange travel.",
    "Subject relocated to be closer to family in another state; nearest site >300 miles away.",
    "Subject started a job with rotating offshore shifts; cannot attend scheduled visits.",
    "Sponsor terminated study participation in this country due to regulatory decision.",
    "Subject moved to Australia; no participating sites in that country.",
    "Site lost IRB approval; subject declined transfer only because no site is within 200 km.",
    "Subject became primary carer for a disabled family member abroad and relocated.",
    "Subject relocated for university studies in another country.",
    "Long-term hospitalization of spouse in another city; subject relocated.",
    "Subject emigrated; site confirmed no participating site in the new country.",
    "Site closed by sponsor for business reasons; subject could not be transferred.",
    "Subject moved to a remote area with no access to a participating site.",
    "Subject relocated for a new job; nearest participating site is in another country.",
]

PROBLEMS = [
    # Misclassified: the free text maps to an existing controlled term
    Scenario("Pt developed severe rash on arms and torso, investigator stopped study drug and pt left study",
             "MISCLASSIFIED_REASON", "Free text describes an adverse event (rash); should be ADVERSE EVENT.",
             overrides={"rash_ae": True}),
    Scenario("Unable to contact subject after 3 phone calls and a certified letter",
             "MISCLASSIFIED_REASON", "Describes lost to follow-up; should be LOST TO FOLLOW-UP."),
    Scenario("Subject withdrew consent",
             "MISCLASSIFIED_REASON", "Consent withdrawal; should be WITHDRAWAL BY SUBJECT."),
    Scenario("Subject no longer wishes to participate in the study",
             "MISCLASSIFIED_REASON", "Subject's choice; should be WITHDRAWAL BY SUBJECT."),
    Scenario("Disease progression, drug not working per PI",
             "MISCLASSIFIED_REASON", "Describes lack of efficacy; should be LACK OF EFFICACY."),
    Scenario("PI decided to remove subject from study for safety reasons",
             "MISCLASSIFIED_REASON", "Investigator decision; should be PHYSICIAN DECISION."),
    Scenario("Subject did not meet inclusion criterion #4 (eGFR), discovered after randomization",
             "MISCLASSIFIED_REASON", "Eligibility violation; should be PROTOCOL DEVIATION."),
    # Conflicts with other columns
    Scenario("Subject passed away at home",
             "CONFLICTS_WITH_DATA", "Text reports death but DTHFL=N and DTHDT is blank; reason should be DEATH.",
             overrides={"alive_after": True}),
    Scenario("Pregnancy confirmed at week 12 visit, discontinued per protocol",
             "CONFLICTS_WITH_DATA", "Pregnancy reported for a subject with SEX=M.",
             overrides={"sex_m": True}),
    Scenario("Discontinued due to ongoing liver enzyme elevations",
             "CONFLICTS_WITH_DATA", "Text cites liver enzyme AE but ADAE has no hepatic events (only headache); "
             "also arguably ADVERSE EVENT.",
             overrides={"no_liver_ae": True}),
    Scenario("Subject moved abroad on {future_date}",
             "CONFLICTS_WITH_DATA", "Date in free text is after EOSDT.",
             overrides={"future_date": True}),
    Scenario("Completed all study visits per protocol",
             "CONFLICTS_WITH_DATA", "Text says completed but EOSSTT=DISCONTINUED and study ended early."),
    Scenario("Subject never received study drug",
             "CONFLICTS_WITH_DATA", "TRTSDT populated and SAFFL=Y, so the subject was dosed."),
    # Unintelligible / uninformative
    Scenario("N/A", "UNINTELLIGIBLE_OR_UNINFORMATIVE", "Non-informative specify text."),
    Scenario("see source", "UNINTELLIGIBLE_OR_UNINFORMATIVE", "Refers to source instead of specifying reason."),
    Scenario("asdf", "UNINTELLIGIBLE_OR_UNINFORMATIVE", "Keyboard mash."),
    Scenario("other", "UNINTELLIGIBLE_OR_UNINFORMATIVE", "Repeats 'other' without specifying."),
    # PII / inappropriate content
    Scenario("Spoke with daughter Maria Lopez (555-212-4478), subject is moving to Florida",
             "PII_OR_INAPPROPRIATE", "Contains a third party's name and phone number."),
    Scenario("John R. Whitfield asked to stop, says he is too busy with work",
             "PII_OR_INAPPROPRIATE", "Contains subject's full name; also describes WITHDRAWAL BY SUBJECT."),
    # Specify text populated although the reason is not OTHER
    Scenario("Subject relocated to another state",
             "CONFLICTS_WITH_DATA", "DCSREAS=ADVERSE EVENT but specify text describes relocation; no AE leads to "
             "withdrawal.", dcsreas="ADVERSE EVENT"),
]

LEGIT_NON_OTHER = [
    Scenario("Moved, new address and phone number unknown; 3 contact attempts documented",
             "NONE", "Specify text consistent with LOST TO FOLLOW-UP.", dcsreas="LOST TO FOLLOW-UP"),
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def make_ae(rng, usubjid, start: date, end: date, term=None, acn=None, sev=None) -> dict:
    decod, bodsys = term or AE_TERMS[int(rng.integers(len(AE_TERMS)))]
    astdt = rand_date(rng, start, max(start, end))
    sev = sev or rng.choice(["MILD", "MODERATE", "SEVERE"], p=[0.6, 0.32, 0.08])
    aendt = astdt + timedelta(days=int(rng.integers(2, 30)))
    return {
        "USUBJID": usubjid,
        "AEDECOD": decod,
        "AEBODSYS": bodsys,
        "AESEV": sev,
        "AESER": "Y" if sev == "SEVERE" and rng.random() < 0.3 else "N",
        "AEREL": rng.choice(["RELATED", "NOT RELATED"], p=[0.4, 0.6]),
        "ASTDT": astdt,
        "AENDT": aendt if aendt <= DATA_CUTOFF else None,
        "AEACN": acn or rng.choice(
            ["DOSE NOT CHANGED", "DOSE REDUCED", "DRUG INTERRUPTED"], p=[0.85, 0.08, 0.07]
        ),
        "TRTEMFL": "Y",
    }


def generate_subjects(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    subjects, aes = [], []
    site_ids = list(SITES)
    # Enrollment ramps up: later dates are more likely.
    span = (LAST_RAND - FIRST_RAND).days
    offsets = np.sort((rng.beta(1.4, 1.0, N_SUBJECTS) * span).astype(int))
    site_counters = {s: 0 for s in site_ids}

    for i, off in enumerate(offsets):
        siteid = site_ids[int(rng.integers(len(site_ids)))]
        site_counters[siteid] += 1
        subjid = f"{siteid}-{site_counters[siteid]:03d}"
        usubjid = f"{STUDYID}-{subjid}"
        armcd, arm, p_ae = ARMS[i % 3]
        sex = rng.choice(["F", "M"])
        age = int(np.clip(rng.normal(56, 12), 18, 85))
        randdt = FIRST_RAND + timedelta(days=int(off))
        trtsdt = randdt + timedelta(days=int(rng.integers(0, 4)))
        planned_trtedt = trtsdt + timedelta(days=TREATMENT_DAYS)

        # Treatment discontinuation
        p_ae_site = min(p_ae * HOT_SITES.get(siteid, 1.0), 0.6)
        reasons = {"ADVERSE EVENT": p_ae_site, **OTHER_DISC, "OTHER": 0.01}
        draw, cum, dctreas = rng.random(), 0.0, ""
        for reason, p in reasons.items():
            cum += p
            if draw < cum:
                dctreas = reason
                break

        disc_date = trtsdt + timedelta(days=int(rng.integers(14, TREATMENT_DAYS - 10)))
        if dctreas and disc_date > DATA_CUTOFF:
            dctreas = ""  # would discontinue later; still on treatment at cutoff

        dctreasp = ""
        if dctreas:
            eotstt, trtedt = "DISCONTINUED", disc_date
            if dctreas == "OTHER":
                dctreasp = str(rng.choice(LEGIT_TRT_OTHER))
        elif planned_trtedt <= DATA_CUTOFF:
            eotstt, trtedt = "COMPLETED", planned_trtedt
        else:
            eotstt, trtedt = "ONGOING", None

        # Study status
        eosstt, eosdt, dcsreas = "ONGOING", None, ""
        # Subjects stopping treatment for OTHER reasons stay in follow-up; the only study-level OTHER
        # discontinuations are the planted specify cases below.
        leaves_study = dctreas in {"WITHDRAWAL BY SUBJECT", "LOST TO FOLLOW-UP", "DEATH"} or (
            dctreas not in {"", "OTHER"} and rng.random() < 0.4
        )
        if dctreas and leaves_study:
            eosstt, dcsreas = "DISCONTINUED", dctreas
            eosdt = trtedt + timedelta(days=int(rng.integers(0, 15))) if dctreas != "DEATH" else trtedt
            eosdt = min(eosdt, DATA_CUTOFF)
        else:
            study_end = trtsdt + timedelta(days=TREATMENT_DAYS + FOLLOW_UP_DAYS)
            if study_end <= DATA_CUTOFF:
                eosstt, eosdt = "COMPLETED", study_end

        dthfl = "Y" if dctreas == "DEATH" else "N"
        dthdt = eosdt if dthfl == "Y" else None
        lstalvdt = eosdt if eosdt else DATA_CUTOFF - timedelta(days=int(rng.integers(1, 40)))

        subjects.append({
            "STUDYID": STUDYID,
            "USUBJID": usubjid,
            "SUBJID": subjid,
            "SITEID": siteid,
            "COUNTRY": SITES[siteid],
            "AGE": age,
            "AGEGR1": ">=65" if age >= 65 else "<65",
            "SEX": sex,
            "RACE": rng.choice(
                ["WHITE", "BLACK OR AFRICAN AMERICAN", "ASIAN", "AMERICAN INDIAN OR ALASKA NATIVE", "OTHER"],
                p=[0.62, 0.14, 0.18, 0.02, 0.04],
            ),
            "ARM": arm,
            "ARMCD": armcd,
            "TRT01P": arm,
            "TRT01A": arm,
            "RANDDT": randdt,
            "TRTSDT": trtsdt,
            "TRTEDT": trtedt,
            "ITTFL": "Y",
            "SAFFL": "Y",
            "EOTSTT": eotstt,
            "DCTREAS": dctreas,
            "DCTREASP": dctreasp,
            "EOSSTT": eosstt,
            "EOSDT": eosdt,
            "DCSREAS": dcsreas,
            "DCSREASP": "",
            "DTHFL": dthfl,
            "DTHDT": dthdt,
            "LSTALVDT": lstalvdt,
        })

        # Adverse events
        window_end = min(trtedt or DATA_CUTOFF, DATA_CUTOFF)
        n_ae = int(rng.poisson({"PBO": 1.2, "D50": 1.8, "D100": 2.4}[armcd]))
        for _ in range(n_ae):
            aes.append(make_ae(rng, usubjid, trtsdt, window_end))
        if dctreas == "ADVERSE EVENT":
            term = AE_DISC_TERMS[int(rng.integers(len(AE_DISC_TERMS)))]
            onset = max(trtsdt, trtedt - timedelta(days=int(rng.integers(1, 10))))
            aes.append(make_ae(rng, usubjid, onset, onset, term=term, acn="DRUG WITHDRAWN",
                               sev=rng.choice(["MODERATE", "SEVERE"], p=[0.6, 0.4])))

    return subjects, aes


def apply_scenario(rng, subj: dict, aes: list[dict], sc: Scenario) -> None:
    """Turn an on-treatment subject into a study discontinuation with the planted specify text."""
    usubjid = subj["USUBJID"]
    disc = subj["TRTSDT"] + timedelta(days=int(rng.integers(30, 200)))
    disc = min(disc, DATA_CUTOFF - timedelta(days=20))
    text = sc.text

    subj.update({
        "EOTSTT": "DISCONTINUED",
        "TRTEDT": disc,
        "DCTREAS": sc.dcsreas,
        "DCTREASP": "",
        "EOSSTT": "DISCONTINUED",
        "EOSDT": disc + timedelta(days=int(rng.integers(1, 10))),
        "DCSREAS": sc.dcsreas,
        "DTHFL": "N",
        "DTHDT": None,
    })
    subj["LSTALVDT"] = subj["EOSDT"]

    # Drop AEs after the new treatment end; keep the subject's AE history otherwise.
    aes[:] = [a for a in aes if a["USUBJID"] != usubjid or a["ASTDT"] <= disc]

    o = sc.overrides
    if o.get("rash_ae"):
        subj["DCTREAS"] = "ADVERSE EVENT"
        aes.append(make_ae(rng, usubjid, disc - timedelta(days=3), disc - timedelta(days=3),
                           term=("Rash", "Skin and subcutaneous tissue disorders"),
                           acn="DRUG WITHDRAWN", sev="SEVERE"))
    if o.get("alive_after"):
        subj["LSTALVDT"] = subj["EOSDT"] + timedelta(days=12)
    if o.get("sex_m"):
        subj["SEX"] = "M"
    if o.get("no_liver_ae"):
        aes[:] = [a for a in aes if a["USUBJID"] != usubjid]
        aes.append(make_ae(rng, usubjid, subj["TRTSDT"], disc,
                           term=("Headache", "Nervous system disorders"), sev="MILD",
                           acn="DOSE NOT CHANGED"))
    if o.get("future_date"):
        text = text.format(future_date=iso(subj["EOSDT"] + timedelta(days=90)))
    if sc.dcsreas == "OTHER":
        subj["DCTREASP"] = text

    subj["DCSREASP"] = text


def main() -> None:
    rng = np.random.default_rng(SEED)
    subjects, aes = generate_subjects(rng)

    # Candidates for planted scenarios: on treatment >= 45 days before cutoff, no discontinuation yet.
    candidates = [
        i for i, s in enumerate(subjects)
        if not s["DCTREAS"] and s["EOSSTT"] != "COMPLETED"
        and (DATA_CUTOFF - s["TRTSDT"]).days > 60
    ]
    scenarios = [Scenario(t, "NONE", "Legitimate 'Other' reason not covered by controlled terms.")
                 for t in LEGIT] + PROBLEMS + LEGIT_NON_OTHER
    chosen = rng.choice(candidates, size=len(scenarios), replace=False)

    key_rows = []
    for idx, sc in zip(chosen, scenarios):
        subj = subjects[int(idx)]
        apply_scenario(rng, subj, aes, sc)
        key_rows.append({
            "USUBJID": subj["USUBJID"],
            "DCSREAS": subj["DCSREAS"],
            "DCSREASP": subj["DCSREASP"],
            "EXPECTED_FLAG": sc.category != "NONE",
            "PLANTED_CATEGORY": sc.category,
            "DESCRIPTION": sc.description,
        })

    adsl = pd.DataFrame(subjects).sort_values("USUBJID").reset_index(drop=True)
    for col in ["RANDDT", "TRTSDT", "TRTEDT", "EOSDT", "DTHDT", "LSTALVDT"]:
        adsl[col] = adsl[col].map(iso)

    adae = pd.DataFrame(aes).sort_values(["USUBJID", "ASTDT"]).reset_index(drop=True)
    adae.insert(1, "AESEQ", adae.groupby("USUBJID").cumcount() + 1)
    adae.insert(0, "STUDYID", STUDYID)
    for col in ["ASTDT", "AENDT"]:
        adae[col] = adae[col].map(iso)

    key = pd.DataFrame(key_rows).sort_values("USUBJID")

    adsl.to_csv(OUT_DIR / "adsl.csv", index=False)
    adae.to_csv(OUT_DIR / "adae.csv", index=False)
    key.to_csv(OUT_DIR / "ai_seeded_anomalies.csv", index=False)

    n_ae_disc = (adsl["DCTREAS"] == "ADVERSE EVENT").sum()
    print(f"ADSL: {len(adsl)} subjects | ADAE: {len(adae)} events")
    print(f"AE treatment discontinuations: {n_ae_disc} ({n_ae_disc / len(adsl):.1%})")
    print(f"Planted specify records: {len(key)} ({key['EXPECTED_FLAG'].sum()} problematic)")


if __name__ == "__main__":
    main()
