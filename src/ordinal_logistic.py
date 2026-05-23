# Ordinal logistic regression of curriculum maturity (exploratory).
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config
from data_loader import build_university_features, compute_maturity, MATURITY_LABELS
from classification_loader import _load_rawdata_curriculum, load_university_info

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT = _SCRIPTS_DIR.parent
ACTIVE_CLASSIFICATION = config.CLASSIFICATION_DIR / "final_adjudicated_classification.csv"
OUT_DIR = config.STATISTICS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_active_course_df() -> pd.DataFrame:
    raw = _load_rawdata_curriculum()
    cls = pd.read_csv(ACTIVE_CLASSIFICATION, encoding="utf-8-sig")

    short_to_full = {
        "D1": "D1_Quantitative_Foundations",
        "D2": "D2_AI_ML",
        "D3": "D3_Data_Science",
        "D4": "D4_Health_Informatics",
        "D5": "D5_Clinical_AI_Application",
    }
    cls = cls[["Course_ID", "D1", "D2", "D3", "D4", "D5"]].copy()
    merged = raw.merge(cls, on="Course_ID", how="inner", validate="one_to_one")
    for short, full in short_to_full.items():
        merged[full] = merged[short].astype(int)
    merged = merged.drop(columns=["D1", "D2", "D3", "D4", "D5"])
    merged.attrs["classification_source"] = "final_adjudicated"
    return merged


def fit_ordinal_logistic(uni_df: pd.DataFrame) -> dict:
    df = uni_df.dropna(subset=["Admission_Quota"]).copy()
    df["Maturity_Level"] = df["Maturity_Level"].astype(float)

    df = df[df["Maturity_Level"] >= 1].copy()

    college_dum = pd.get_dummies(df["College"], drop_first=True, dtype=float)
    X = pd.concat(
        [college_dum, df[["Is_Public", "Admission_Quota"]].astype(float)],
        axis=1,
    )
    y = df["Maturity_Level"]

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    mod = OrderedModel(y, X, distr="logit")
    res = mod.fit(method="bfgs", disp=False)

    rows = []
    for name, coef, se, p in zip(res.params.index, res.params.values, res.bse.values, res.pvalues.values):
        or_point = float(np.exp(coef))
        or_low = float(np.exp(coef - 1.96 * se))
        or_high = float(np.exp(coef + 1.96 * se))
        rows.append(
            {
                "Variable": name,
                "Coefficient": float(coef),
                "SE": float(se),
                "p_value": float(p),
                "OR": or_point,
                "OR_95CI_low": or_low,
                "OR_95CI_high": or_high,
            }
        )
    coef_df = pd.DataFrame(rows)

    summary = {
        "n": int(len(df)),
        "outcome_levels": sorted(df["Maturity_Level"].unique().tolist()),
        "outcome_label_map": {1: "Foundational-Only", 2: "Intermediate", 3: "Advanced"},
        "predictors": list(X.columns),
        "reference_levels": {
            "College": "Medicine",
            "Is_Public": "0 = Private",
        },
        "log_likelihood": float(res.llf),
        "log_likelihood_null": float(res.llnull) if hasattr(res, "llnull") else None,
        "mcfadden_pseudo_R2": float(res.prsquared) if hasattr(res, "prsquared") else None,
        "aic": float(res.aic) if hasattr(res, "aic") else None,
        "bic": float(res.bic) if hasattr(res, "bic") else None,
        "converged": bool(
            res.mle_retvals.get("converged", True) if hasattr(res, "mle_retvals") else True
        ),
        "classification_source": "final_adjudicated",
        "active_data_file": str(ACTIVE_CLASSIFICATION.relative_to(PROJECT)).replace("\\", "/"),
    }
    return {"coef_df": coef_df, "summary": summary}


def main() -> None:
    course_df = load_active_course_df()
    uni_info = load_university_info()
    uni_df = build_university_features(course_df, uni_info)
    uni_df["Maturity_Level"] = compute_maturity(uni_df)
    uni_df["Maturity_Label"] = uni_df["Maturity_Level"].map(MATURITY_LABELS)
    print(f"[ordinal_logistic] N schools = {len(uni_df)}")
    print(f"  Maturity distribution: {uni_df['Maturity_Label'].value_counts().to_dict()}")

    out = fit_ordinal_logistic(uni_df)
    coef_csv = OUT_DIR / "ordinal_logistic.csv"
    summary_json = OUT_DIR / "ordinal_logistic_summary.json"

    out["coef_df"].to_csv(coef_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(out["summary"], f, indent=2, ensure_ascii=False)

    print(f"[ordinal_logistic] Wrote {coef_csv.relative_to(PROJECT)}")
    print(f"[ordinal_logistic] Wrote {summary_json.relative_to(PROJECT)}")
    print()
    print("=== Coefficient table ===")
    print(out["coef_df"].to_string(index=False))
    print()
    print("=== Summary ===")
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
