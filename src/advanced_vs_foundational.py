# Advanced vs Foundational-Only extreme-group profiling.
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config
from data_loader import build_university_features, compute_maturity, MATURITY_LABELS
from classification_loader import load_university_info
from ordinal_logistic import load_active_course_df

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT = _SCRIPTS_DIR.parent
ACTIVE_CLASSIFICATION = config.CLASSIFICATION_DIR / "final_adjudicated_classification.csv"
OUT_DIR = config.STATISTICS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    return abs(1.0 - 2.0 * u / (n1 * n2))


def median_iqr(series: pd.Series) -> tuple[float, float, float]:
    s = series.dropna()
    return float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))


def compare_one(name: str, adv_vals: pd.Series, found_vals: pd.Series) -> dict:
    u, p = stats.mannwhitneyu(adv_vals.values, found_vals.values, alternative="two-sided")
    r = rank_biserial_from_u(float(u), len(adv_vals), len(found_vals))
    adv_med, adv_q1, adv_q3 = median_iqr(adv_vals)
    fnd_med, fnd_q1, fnd_q3 = median_iqr(found_vals)
    return {
        "Variable": name,
        "Adv_n": int(len(adv_vals)),
        "Adv_median": adv_med,
        "Adv_IQR_low": adv_q1,
        "Adv_IQR_high": adv_q3,
        "Foundational_n": int(len(found_vals)),
        "Foundational_median": fnd_med,
        "Foundational_IQR_low": fnd_q1,
        "Foundational_IQR_high": fnd_q3,
        "U": float(u),
        "p_value": float(p),
        "abs_rank_biserial_r": float(r),
    }


def main() -> None:
    course_df = load_active_course_df()
    uni_info = load_university_info()
    uni_df = build_university_features(course_df, uni_info)
    uni_df["Maturity_Level"] = compute_maturity(uni_df)
    uni_df["Maturity_Label"] = uni_df["Maturity_Level"].map(MATURITY_LABELS)

    adv = uni_df[uni_df["Maturity_Label"] == "Advanced"].copy()
    fnd = uni_df[uni_df["Maturity_Label"] == "Foundational-Only"].copy()
    print(f"[adv_vs_foundational] Advanced n = {len(adv)}, Foundational-Only n = {len(fnd)}")

    rows = []
    rows.append(compare_one("Total credits", adv["Total_Credits"], fnd["Total_Credits"]))
    rows.append(compare_one("Total courses", adv["Total_Courses"], fnd["Total_Courses"]))
    rows.append(compare_one("Credits per course", adv["Credits_per_Course"], fnd["Credits_per_Course"]))
    rows.append(compare_one("Breadth (domains with ≥1 course)", adv["Breadth"], fnd["Breadth"]))
    rows.append(compare_one("Mandatory ratio", adv["Mandatory_Ratio"], fnd["Mandatory_Ratio"]))
    rows.append(compare_one("Pre-medical ratio", adv["Pre_Medical_Ratio"], fnd["Pre_Medical_Ratio"]))
    rows.append(compare_one("Admission quota (seats)", adv["Admission_Quota"].dropna(), fnd["Admission_Quota"].dropna()))
    rows.append(compare_one("Is_Public (proportion)", adv["Is_Public"], fnd["Is_Public"]))

    df_out = pd.DataFrame(rows)
    csv_out = OUT_DIR / "adv_vs_foundational.csv"
    df_out.to_csv(csv_out, index=False, encoding="utf-8-sig")

    summary = {
        "n_advanced": int(len(adv)),
        "n_foundational_only": int(len(fnd)),
        "test": "Mann-Whitney U, two-sided",
        "effect_size": "abs(rank-biserial r) = |1 - 2U/(n1*n2)|",
        "multiplicity_correction": "None (extreme-group profiling, not confirmatory testing)",
        "classification_source": "final_adjudicated",
        "active_data_file": str(ACTIVE_CLASSIFICATION.relative_to(PROJECT)).replace("\\", "/"),
    }
    json_out = OUT_DIR / "adv_vs_foundational_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[adv_vs_foundational] Wrote {csv_out.relative_to(PROJECT)}")
    print(f"[adv_vs_foundational] Wrote {json_out.relative_to(PROJECT)}")
    print()
    print("=== Comparison table ===")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
