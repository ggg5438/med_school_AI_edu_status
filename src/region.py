# -*- coding: utf-8 -*-
"""
Capital vs non-Capital comparison on the full 63-school cohort.

Two-sided Mann-Whitney U on the continuous family below, with Holm-Bonferroni
correction applied within the family. Capital area = Region 'Seoul/Gyeonggi'
(18 schools) vs the remaining 45.

Reuses the loaders and constants from analysis.py.
Output: results/statistics/region_mwu.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
import analysis as va  # reuse loaders + constants

ROOT = _SRC.parent
OUT = ROOT / "results" / "statistics"
OUT.mkdir(parents=True, exist_ok=True)

# Continuous family; Holm correction is applied within this family.
FAMILY = [
    "Total_Credits",
    "Mandatory_Ratio",
    "Pre_Medical_Ratio",
    "D1_credits",
    "D2_credits",
    "D3_credits",
    "D4_credits",
    "D5_credits",
]


def build_region_features():
    meta = va.load_university_info()
    cls = va.load_classification(va.CLS_PATH)
    year = va.load_year_metadata()
    clsy = cls.merge(year, on="Course_ID", how="left")
    rows = []
    for _, m in meta.iterrows():
        uni, coll = m["University"], m["College"]
        sub = clsy[(clsy["SchoolUni"] == uni) & (clsy["SchoolColl"] == coll)]
        n = len(sub)
        row = {
            "University": uni, "College": coll,
            "Capital_Area": int(m["Region"] in va.CAPITAL_REGIONS),
            "Total_Credits": float(sub["Credits"].sum()),
            "Mandatory_Ratio": float((sub["Is_Mandatory"] == 1).mean()) if n else 0.0,
            "Pre_Medical_Ratio": float((sub["Pre_Medical"] == 1).mean()) if n else 0.0,
        }
        for d in va.DOMAINS:
            row[f"{d}_credits"] = float(sub.loc[sub[d] == 1, "Credits"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = build_region_features()
    cap = df[df["Capital_Area"] == 1]
    non = df[df["Capital_Area"] == 0]
    assert len(cap) == 18 and len(non) == 45, f"split {len(cap)}/{len(non)} != 18/45"

    rows = []
    for var in FAMILY:
        a, b = cap[var].values, non[var].values
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r = 1 - (2 * u) / (len(a) * len(b))
        rows.append({
            "Variable": var,
            "Capital_median": float(np.median(a)), "NonCapital_median": float(np.median(b)),
            "U": float(u), "p_value": float(p), "rank_biserial_r": float(r),
        })
    res = pd.DataFrame(rows)
    # Holm within family
    pvals = res["p_value"].values
    order = np.argsort(pvals)
    m = len(pvals)
    holm = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(pvals[idx] * (m - rank), 1.0))
        holm[idx] = run
    res["p_holm"] = holm
    res.to_csv(OUT / "region_mwu.csv", index=False, encoding="utf-8-sig")

    print(f"Capital n={len(cap)}, Non-Capital n={len(non)}; Holm family size m={m}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
