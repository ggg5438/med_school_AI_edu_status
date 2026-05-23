# School-type decomposition into four explicit axes.
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config
from classification_loader import load_all

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = config.STATISTICS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def annotate_axes(uni_df: pd.DataFrame) -> pd.DataFrame:
    df = uni_df.copy()
    df["Region_Binary"] = np.where(
        df["Region"] == "Seoul/Gyeonggi", "Seoul", "Non-Seoul"
    )
    df["Governance"] = np.where(df["Is_Public"] == 1, "Public", "Private")
    qcuts = pd.qcut(
        df["Admission_Quota"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    df["Quota_Quartile"] = qcuts.astype(str)
    return df


def axis_summary(uni_df: pd.DataFrame, axis_col: str) -> pd.DataFrame:
    rows = []
    for level, sub in uni_df.groupby(axis_col, observed=True):
        rows.append({
            axis_col: level,
            "N_Schools": len(sub),
            "Total_Credits_Mean": float(sub["Total_Credits"].mean()),
            "Total_Credits_SD": float(sub["Total_Credits"].std(ddof=1)),
            "Total_Credits_Median": float(sub["Total_Credits"].median()),
            "AI_Core_Credits_Mean": float(sub["AI_Core_Credits"].mean()),
            "AI_Core_Credits_SD": float(sub["AI_Core_Credits"].std(ddof=1)),
            "Mandatory_Ratio_Mean": float(sub["Mandatory_Ratio"].mean()),
            "Pct_Advanced": float((sub["Maturity_Level"] == 3).mean() * 100),
            "Pct_Intermediate": float((sub["Maturity_Level"] == 2).mean() * 100),
            "Pct_Foundational_Only": float(
                (sub["Maturity_Level"] == 1).mean() * 100
            ),
            "Pct_Has_AI_Core": float((sub["Has_AI_Core"] == 1).mean() * 100),
        })
    return pd.DataFrame(rows)


def axis_test(uni_df: pd.DataFrame, axis_col: str) -> dict:
    levels = sorted(uni_df[axis_col].unique())
    out = {"axis": axis_col, "levels": levels}
    for var in ["Total_Credits", "AI_Core_Credits"]:
        groups = [uni_df.loc[uni_df[axis_col] == lv, var].values
                  for lv in levels]
        if len(levels) >= 3:
            stat, p = sstats.kruskal(*groups)
            test = "Kruskal-Wallis"
            n = sum(len(g) for g in groups)
            eta_sq_h = (stat - len(levels) + 1) / (n - len(levels))
            es_label = "epsilon_squared_H"
            es_value = float(max(0.0, eta_sq_h))
        else:
            u, p = sstats.mannwhitneyu(
                groups[0], groups[1], alternative="two-sided"
            )
            stat = u
            test = "Mann-Whitney U"
            r = 1 - (2 * u) / (len(groups[0]) * len(groups[1]))
            es_label = "rank_biserial_r"
            es_value = float(r)
        out[f"{var}_test"] = test
        out[f"{var}_statistic"] = float(stat)
        out[f"{var}_p_value"] = float(p)
        out[f"{var}_{es_label}"] = es_value
    mat_groups = [uni_df.loc[uni_df[axis_col] == lv, "Maturity_Level"].values
                  for lv in levels]
    if len(levels) >= 3:
        stat, p = sstats.kruskal(*mat_groups)
        test = "Kruskal-Wallis"
    else:
        stat, p = sstats.mannwhitneyu(
            mat_groups[0], mat_groups[1], alternative="two-sided"
        )
        test = "Mann-Whitney U"
    out["Maturity_Level_test"] = test
    out["Maturity_Level_statistic"] = float(stat)
    out["Maturity_Level_p_value"] = float(p)
    return out


def main() -> int:
    print("=" * 70)
    print("SCHOOL TYPE — 4-axis decomposition")
    print("=" * 70)
    course_df, uni_info, uni_df = load_all(classifier="consensus")
    uni_df = annotate_axes(uni_df)

    axes = {
        "axisA_College": ("College", "school_type_axisA_college.csv"),
        "axisB_Governance": ("Governance", "school_type_axisB_governance.csv"),
        "axisC_Region": ("Region_Binary", "school_type_axisC_region.csv"),
        "axisD_Quota": ("Quota_Quartile", "school_type_axisD_quota.csv"),
    }

    summary = {}
    for key, (col, fname) in axes.items():
        df = axis_summary(uni_df, col)
        df.to_csv(OUT_DIR / fname, index=False, encoding="utf-8-sig")
        test = axis_test(uni_df, col)
        summary[key] = test
        print(f"\n[{key}]")
        print(df.to_string(index=False))
        print(f"  Test: {test}")

    q_cuts = uni_df.groupby("Quota_Quartile")["Admission_Quota"].agg(
        ["min", "max", "count"]
    )
    summary["Quota_Quartile_Cutoffs"] = q_cuts.reset_index().to_dict("records")

    with open(OUT_DIR / "school_type_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {OUT_DIR / 'school_type_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
