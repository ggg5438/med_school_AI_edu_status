# Region analysis: Capital area vs Non-Capital area.
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.stats.contingency import odds_ratio as scipy_odds_ratio

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config
from classification_loader import load_all

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = config.STATISTICS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 1000
CONTINUOUS_VARS = [
    "Total_Credits",
    "AI_Core_Credits",
    "Mandatory_Ratio",
    "Pre_Medical_Ratio",
    "D1_Quantitative_Foundations_Credits",
    "D2_AI_ML_Credits",
    "D3_Data_Science_Credits",
    "D4_Health_Informatics_Credits",
    "D5_Clinical_AI_Application_Credits",
]


def assign_region_binary(uni_df: pd.DataFrame) -> pd.DataFrame:
    uni_df = uni_df.copy()
    uni_df["Region_Binary"] = np.where(
        uni_df["Region"] == "Seoul/Gyeonggi", "Seoul", "Non-Seoul"
    )
    return uni_df


def descriptive(uni_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in ["Seoul", "Non-Seoul"]:
        sub = uni_df[uni_df["Region_Binary"] == region]
        rows.append({
            "Region": region,
            "N_Schools": len(sub),
            "N_Medicine": int((sub["College"] == "Medicine").sum()),
            "N_Dentistry": int((sub["College"] == "Dentistry").sum()),
            "N_Korean_Medicine": int((sub["College"] == "Korean Medicine").sum()),
            "N_Public": int((sub["Is_Public"] == 1).sum()),
            "N_Private": int((sub["Is_Public"] == 0).sum()),
            "Total_Credits_Mean": float(sub["Total_Credits"].mean()),
            "Total_Credits_SD": float(sub["Total_Credits"].std(ddof=1)),
            "Total_Credits_Median": float(sub["Total_Credits"].median()),
            "AI_Core_Credits_Mean": float(sub["AI_Core_Credits"].mean()),
            "AI_Core_Credits_SD": float(sub["AI_Core_Credits"].std(ddof=1)),
            "Mandatory_Ratio_Mean": float(sub["Mandatory_Ratio"].mean()),
            "Mandatory_Ratio_SD": float(sub["Mandatory_Ratio"].std(ddof=1)),
            "Pct_Advanced": float((sub["Maturity_Level"] == 3).mean() * 100),
            "Pct_Has_AI_Core": float((sub["Has_AI_Core"] == 1).mean() * 100),
            "Pct_Has_D2_AI_ML": float((sub["Has_D2_AI_ML"] == 1).mean() * 100),
        })
    return pd.DataFrame(rows)


def mann_whitney_table(uni_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seoul = uni_df[uni_df["Region_Binary"] == "Seoul"]
    nonseoul = uni_df[uni_df["Region_Binary"] == "Non-Seoul"]
    for var in CONTINUOUS_VARS:
        s_vals = seoul[var].values
        n_vals = nonseoul[var].values
        u_stat, p_two = sstats.mannwhitneyu(
            s_vals, n_vals, alternative="two-sided"
        )
        n1 = len(s_vals)
        n2 = len(n_vals)
        r_signed = 1 - (2 * u_stat) / (n1 * n2)
        rows.append({
            "Variable": var,
            "Seoul_Median": float(np.median(s_vals)),
            "Seoul_IQR_Low": float(np.percentile(s_vals, 25)),
            "Seoul_IQR_High": float(np.percentile(s_vals, 75)),
            "NonSeoul_Median": float(np.median(n_vals)),
            "NonSeoul_IQR_Low": float(np.percentile(n_vals, 25)),
            "NonSeoul_IQR_High": float(np.percentile(n_vals, 75)),
            "U_Statistic": float(u_stat),
            "p_value": float(p_two),
            "rank_biserial_r": float(r_signed),
            "abs_rank_biserial_r": float(abs(r_signed)),
        })
    df = pd.DataFrame(rows)
    pvals = df["p_value"].values
    order = np.argsort(pvals)
    m = len(pvals)
    holm = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvals[idx] * (m - rank)
        running_max = max(running_max, min(adj, 1.0))
        holm[idx] = running_max
    df["p_holm"] = holm
    return df


def fisher_exact_block(uni_df: pd.DataFrame) -> dict:
    out = {}
    for outcome_col, label in [
        ("Maturity_Level_3", "Advanced_x_Region"),
        ("Has_AI_Core", "AnyAICore_x_Region"),
        ("Has_D2_AI_ML", "D2_AI_ML_x_Region"),
        ("Has_D5_Clinical_AI_Application", "D5_ClinicalAI_x_Region"),
    ]:
        if outcome_col == "Maturity_Level_3":
            outcome = (uni_df["Maturity_Level"] == 3).astype(int)
        else:
            outcome = uni_df[outcome_col]
        seoul_mask = uni_df["Region_Binary"] == "Seoul"
        a = int(((outcome == 1) & seoul_mask).sum())
        b = int(((outcome == 0) & seoul_mask).sum())
        c = int(((outcome == 1) & ~seoul_mask).sum())
        d = int(((outcome == 0) & ~seoul_mask).sum())
        ct = np.array([[a, b], [c, d]])
        OR_sample, p = sstats.fisher_exact(ct, alternative="two-sided")
        try:
            r = scipy_odds_ratio(ct, kind="conditional")
            ci = r.confidence_interval(confidence_level=0.95)
            ci_low, ci_high = float(ci.low), float(ci.high)
            or_cmle = float(r.statistic)
        except Exception:
            ci_low, ci_high, or_cmle = float("nan"), float("nan"), float("nan")
        out[label] = {
            "table": {
                "Seoul_outcome1": a,
                "Seoul_outcome0": b,
                "NonSeoul_outcome1": c,
                "NonSeoul_outcome0": d,
            },
            "odds_ratio_sample": float(OR_sample),
            "odds_ratio_conditional_mle": or_cmle,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "p_value": float(p),
            "reference": "Non-Seoul",
        }
    return out


def _resample_indices(uni_df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    chunks = []
    for college, idx in uni_df.groupby("College").indices.items():
        chunks.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(chunks)


def _bca_ci(theta_boot: np.ndarray, theta_hat: float, jackknife: np.ndarray,
            alpha: float = 0.05) -> tuple[float, float]:
    valid = ~np.isnan(theta_boot)
    theta_boot = theta_boot[valid]
    if len(theta_boot) == 0:
        return float("nan"), float("nan")
    p_less = np.mean(theta_boot < theta_hat)
    if p_less <= 0 or p_less >= 1:
        z0 = 0.0
    else:
        z0 = sstats.norm.ppf(p_less)
    jk_mean = np.mean(jackknife)
    num = np.sum((jk_mean - jackknife) ** 3)
    den = 6 * (np.sum((jk_mean - jackknife) ** 2)) ** 1.5
    a_acc = num / den if den != 0 else 0.0
    z_lo = sstats.norm.ppf(alpha / 2)
    z_hi = sstats.norm.ppf(1 - alpha / 2)
    alpha_lo = sstats.norm.cdf(
        z0 + (z0 + z_lo) / (1 - a_acc * (z0 + z_lo))
    )
    alpha_hi = sstats.norm.cdf(
        z0 + (z0 + z_hi) / (1 - a_acc * (z0 + z_hi))
    )
    lo = float(np.percentile(theta_boot, 100 * alpha_lo))
    hi = float(np.percentile(theta_boot, 100 * alpha_hi))
    return lo, hi


def bca_difference_in_means(uni_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    n_uni = len(uni_df)
    for var in CONTINUOUS_VARS:
        seoul = uni_df.loc[uni_df["Region_Binary"] == "Seoul", var].values
        nons = uni_df.loc[uni_df["Region_Binary"] == "Non-Seoul", var].values
        theta_hat = float(np.mean(seoul) - np.mean(nons))
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = _resample_indices(uni_df, rng)
            sub = uni_df.iloc[idx]
            s = sub.loc[sub["Region_Binary"] == "Seoul", var].values
            n = sub.loc[sub["Region_Binary"] == "Non-Seoul", var].values
            if len(s) == 0 or len(n) == 0:
                boot[b] = np.nan
            else:
                boot[b] = float(np.mean(s) - np.mean(n))
        jk = np.empty(n_uni)
        for i in range(n_uni):
            sub = uni_df.drop(uni_df.index[i])
            s = sub.loc[sub["Region_Binary"] == "Seoul", var].values
            n = sub.loc[sub["Region_Binary"] == "Non-Seoul", var].values
            if len(s) == 0 or len(n) == 0:
                jk[i] = theta_hat
            else:
                jk[i] = float(np.mean(s) - np.mean(n))
        ci_lo, ci_hi = _bca_ci(boot, theta_hat, jk)
        rows.append({
            "Variable": var,
            "Diff_Seoul_minus_NonSeoul": theta_hat,
            "BCa_95_low": ci_lo,
            "BCa_95_high": ci_hi,
            "n_boot": N_BOOT,
            "seed": SEED,
        })
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 70)
    print("REGION ANALYSIS — Capital area vs Non-Capital area")
    print("=" * 70)

    course_df, uni_info, uni_df = load_all(classifier="consensus")
    uni_df = assign_region_binary(uni_df)

    desc = descriptive(uni_df)
    desc.to_csv(OUT_DIR / "region_descriptive.csv", index=False,
                encoding="utf-8-sig")
    print("\n[Descriptive]")
    print(desc.to_string(index=False))

    mwu = mann_whitney_table(uni_df)
    mwu.to_csv(OUT_DIR / "region_mwu.csv", index=False,
               encoding="utf-8-sig")
    print("\n[MWU]")
    print(mwu.to_string(index=False))

    fisher = fisher_exact_block(uni_df)
    with open(OUT_DIR / "region_fisher.json", "w", encoding="utf-8") as f:
        json.dump(fisher, f, indent=2, ensure_ascii=False)
    print("\n[Fisher]")
    for k, v in fisher.items():
        print(f"  {k}: OR={v['odds_ratio_sample']:.2f} "
              f"(95% CI {v['ci_95_low']:.2f}–{v['ci_95_high']:.2f}) "
              f"p={v['p_value']:.4f}")

    bca = bca_difference_in_means(uni_df)
    bca.to_csv(OUT_DIR / "region_bca_ci.csv", index=False,
               encoding="utf-8-sig")
    print("\n[BCa]")
    print(bca.to_string(index=False))

    summary = {
        "n_total": int(len(uni_df)),
        "n_seoul": int((uni_df["Region_Binary"] == "Seoul").sum()),
        "n_nonseoul": int((uni_df["Region_Binary"] == "Non-Seoul").sum()),
        "regions_pooled_into_nonseoul": [
            r for r in uni_df["Region"].unique()
            if r != "Seoul/Gyeonggi"
        ],
        "bootstrap_config": {
            "n_resamples": N_BOOT,
            "seed": SEED,
            "stratification": "College",
        },
        "alpha": 0.05,
        "multiple_comparison": "Holm-Bonferroni within continuous family",
    }
    with open(OUT_DIR / "region_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {OUT_DIR / 'region_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
