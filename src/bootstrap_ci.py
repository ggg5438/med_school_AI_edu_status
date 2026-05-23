# BCa (bias-corrected accelerated) bootstrap 95% confidence intervals.

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats


# Path setup
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config
from classification_loader import load_all

PROJECT_ROOT = _SCRIPTS_DIR.parent
OUT_DIR = config.STATISTICS_DIR
OUT_JSON = OUT_DIR / "bootstrap_bca_effect_sizes.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Bootstrap configuration
N_RESAMPLES = 1000
SEED = int(config.RANDOM_STATE)
CI_ALPHA = 0.05
_Z_HALF = stats.norm.ppf(1 - CI_ALPHA / 2)
_Z_LOW = stats.norm.ppf(CI_ALPHA / 2)


# Point-estimate statistic functions (mirror analysis.py formulas)
def _eta_sq_oneway(values_list: list[np.ndarray]) -> float:
    concat = np.concatenate(values_list)
    if len(concat) == 0:
        return float("nan")
    grand_mean = concat.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in values_list if len(g) > 0)
    ss_total = float(((concat - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return float("nan")
    return float(ss_between / ss_total)


def _cohens_d_two_sample(vals1: np.ndarray, vals2: np.ndarray) -> float:
    n1, n2 = len(vals1), len(vals2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1 = vals1.std(ddof=1) ** 2
    v2 = vals2.std(ddof=1) ** 2
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return float((vals1.mean() - vals2.mean()) / pooled)


def _cramers_v_contingency(table: np.ndarray) -> float:
    try:
        chi2, _, _, _ = stats.chi2_contingency(table)
    except ValueError:
        return float("nan")
    n = float(table.sum())
    k = min(table.shape) - 1
    if n <= 0 or k <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n * k)))


def _rank_biserial_r(vals1: np.ndarray, vals2: np.ndarray) -> float:
    n1, n2 = len(vals1), len(vals2)
    if n1 < 1 or n2 < 1:
        return float("nan")
    try:
        U, _ = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
    except ValueError:
        return float("nan")
    return float(1.0 - (2.0 * U) / (n1 * n2))


# Core BCa engine (stratified)
def _bca_ci_from_samples(
    theta_boot: np.ndarray,
    theta_jack: np.ndarray,
    theta_hat: float,
) -> tuple[float, float, dict]:
    theta_boot = np.asarray(theta_boot, dtype=float)
    finite = np.isfinite(theta_boot)
    theta_boot = theta_boot[finite]
    n_valid = int(finite.sum())

    diag: dict[str, Any] = {
        "n_valid_resamples": n_valid,
        "bootstrap_mean": float(theta_boot.mean()) if n_valid > 0 else float("nan"),
        "bootstrap_sd": float(theta_boot.std(ddof=1)) if n_valid > 1 else float("nan"),
    }

    if n_valid < 10:
        diag["method"] = "insufficient_resamples"
        return float("nan"), float("nan"), diag

    if np.isclose(theta_boot.std(), 0.0):
        diag["method"] = "degenerate_zero_variance"
        diag["degenerate"] = True
        return float(theta_hat), float(theta_hat), diag

    frac_below = float((theta_boot < theta_hat).sum()) / n_valid
    frac_below = min(max(frac_below, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(frac_below)

    theta_jack = np.asarray(theta_jack, dtype=float)
    jf = theta_jack[np.isfinite(theta_jack)]
    if len(jf) < 3:
        diag["method"] = "jackknife_too_small"
        return float("nan"), float("nan"), diag

    jack_mean = jf.mean()
    num = float(((jack_mean - jf) ** 3).sum())
    den = 6.0 * float((((jack_mean - jf) ** 2).sum()) ** 1.5)
    if den == 0 or not np.isfinite(den):
        a = 0.0
    else:
        a = num / den
    if not np.isfinite(a):
        a = 0.0

    def _phi(z_bca: float) -> float:
        denom = 1.0 - a * (z0 + z_bca)
        if denom == 0:
            return float("nan")
        return float(stats.norm.cdf(z0 + (z0 + z_bca) / denom))

    alpha_lo = _phi(_Z_LOW)
    alpha_hi = _phi(_Z_HALF)
    if not np.isfinite(alpha_lo) or not np.isfinite(alpha_hi):
        alpha_lo = CI_ALPHA / 2
        alpha_hi = 1 - CI_ALPHA / 2
        diag["method"] = "bca_fallback_to_percentile"
    else:
        diag["method"] = "bca"

    alpha_lo = min(max(alpha_lo, 1e-6), 1 - 1e-6)
    alpha_hi = min(max(alpha_hi, 1e-6), 1 - 1e-6)

    lo = float(np.quantile(theta_boot, alpha_lo))
    hi = float(np.quantile(theta_boot, alpha_hi))
    diag["z0"] = float(z0)
    diag["acceleration"] = float(a)
    diag["alpha_lo"] = float(alpha_lo)
    diag["alpha_hi"] = float(alpha_hi)
    return lo, hi, diag


def _stratified_indices(
    strata_keys: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    idx_out = []
    for key in np.unique(strata_keys):
        sel = np.where(strata_keys == key)[0]
        if len(sel) == 0:
            continue
        idx_out.append(rng.choice(sel, size=len(sel), replace=True))
    return np.concatenate(idx_out)


def _anova_eta_sq_task(
    uni_df: pd.DataFrame, variable: str
) -> tuple[float, np.ndarray, np.ndarray]:
    colleges = np.array(uni_df["College"].values)
    values = np.array(uni_df[variable].values, dtype=float)
    groups_obs = [values[colleges == c] for c in np.unique(colleges)]
    theta_hat = _eta_sq_oneway(groups_obs)

    rng = np.random.default_rng(SEED)
    boot = np.empty(N_RESAMPLES, dtype=float)
    for b in range(N_RESAMPLES):
        idx = _stratified_indices(colleges, rng)
        vals_b = values[idx]
        col_b = colleges[idx]
        groups_b = [vals_b[col_b == c] for c in np.unique(colleges)]
        boot[b] = _eta_sq_oneway(groups_b)

    n = len(values)
    jack = np.empty(n, dtype=float)
    all_idx = np.arange(n)
    for i in range(n):
        keep = all_idx != i
        v = values[keep]
        c = colleges[keep]
        groups_j = [v[c == k] for k in np.unique(colleges)]
        jack[i] = _eta_sq_oneway(groups_j)

    return theta_hat, boot, jack


def _cohens_d_task(
    uni_df: pd.DataFrame, variable: str
) -> tuple[float, np.ndarray, np.ndarray]:
    is_pub = np.array(uni_df["Is_Public"].values).astype(int)
    values = np.array(uni_df[variable].values, dtype=float)
    pub_obs = values[is_pub == 1]
    priv_obs = values[is_pub == 0]
    theta_hat = _cohens_d_two_sample(pub_obs, priv_obs)

    rng = np.random.default_rng(SEED)
    strata = is_pub
    boot = np.empty(N_RESAMPLES, dtype=float)
    for b in range(N_RESAMPLES):
        idx = _stratified_indices(strata, rng)
        v_b = values[idx]
        s_b = strata[idx]
        boot[b] = _cohens_d_two_sample(v_b[s_b == 1], v_b[s_b == 0])

    n = len(values)
    jack = np.empty(n, dtype=float)
    all_idx = np.arange(n)
    for i in range(n):
        keep = all_idx != i
        v = values[keep]
        s = strata[keep]
        jack[i] = _cohens_d_two_sample(v[s == 1], v[s == 0])

    return theta_hat, boot, jack


def _cramers_v_domain_mandatory_task(
    course_df: pd.DataFrame,
) -> tuple[float, np.ndarray, np.ndarray]:
    domain_cols = list(config.DOMAIN_COLS)

    def _build_table(df_: pd.DataFrame) -> np.ndarray:
        rows = []
        for d in domain_cols:
            sub = df_[df_[d] == 1]
            m = int(sub["Is_Mandatory_Binary"].sum())
            e = int(len(sub) - m)
            rows.append([m, e])
        return np.asarray(rows, dtype=float)

    colleges = np.array(course_df["College"].values)
    tab_obs = _build_table(course_df)
    theta_hat = _cramers_v_contingency(tab_obs)

    rng = np.random.default_rng(SEED)
    boot = np.empty(N_RESAMPLES, dtype=float)
    cdf = course_df.reset_index(drop=True)
    for b in range(N_RESAMPLES):
        idx = _stratified_indices(colleges, rng)
        tab_b = _build_table(cdf.iloc[idx])
        boot[b] = _cramers_v_contingency(tab_b)

    n = len(cdf)
    jack = np.empty(n, dtype=float)
    all_idx = np.arange(n)
    for i in range(n):
        keep = all_idx != i
        tab_j = _build_table(cdf.iloc[keep])
        jack[i] = _cramers_v_contingency(tab_j)

    return theta_hat, boot, jack


def _rank_biserial_task(
    uni_df: pd.DataFrame, variable: str
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    adv_mask = uni_df["Maturity_Level"] == 3
    fo_mask = uni_df["Maturity_Level"] == 1
    sub = uni_df[adv_mask | fo_mask].reset_index(drop=True)
    group = np.where(sub["Maturity_Level"].values == 3, "adv", "fo")
    values = np.asarray(sub[variable].values, dtype=float)
    adv_vals = values[group == "adv"]
    fo_vals = values[group == "fo"]
    theta_hat = _rank_biserial_r(adv_vals, fo_vals)

    rng = np.random.default_rng(SEED)
    boot = np.empty(N_RESAMPLES, dtype=float)
    for b in range(N_RESAMPLES):
        idx = _stratified_indices(group, rng)
        v_b = values[idx]
        g_b = group[idx]
        boot[b] = _rank_biserial_r(v_b[g_b == "adv"], v_b[g_b == "fo"])

    n = len(values)
    jack = np.empty(n, dtype=float)
    all_idx = np.arange(n)
    for i in range(n):
        keep = all_idx != i
        v = values[keep]
        g = group[keep]
        jack[i] = _rank_biserial_r(v[g == "adv"], v[g == "fo"])

    return theta_hat, boot, jack, {"n_adv": int((group == "adv").sum()),
                                   "n_fo": int((group == "fo").sum())}


# Orchestration
def run_all() -> dict:
    t0 = time.time()
    print("=" * 74)
    print("V5 BCa BOOTSTRAP 95% CIs FOR HEADLINE EFFECT SIZES")
    print("=" * 74)
    print(f"Resamples: {N_RESAMPLES}  seed={SEED}  CI: 95% BCa")
    print()

    print("[STEP] Loading data (adjudicated consensus) ...")
    course_df, _, uni_df = load_all(classifier="consensus")

    out: dict[str, dict[str, Any]] = {}
    meta = {
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "method": "BCa (Efron 1987) via jackknife acceleration + "
                  "percentile adjustment from bootstrap bias correction z0",
        "script": "src/bootstrap_ci.py",
        "data_source": {
            "consensus_csv": "data/classification/consensus.csv",
            "course_df_rows": int(len(course_df)),
            "uni_df_rows": int(len(uni_df)),
        },
        "ci_level": 0.95,
    }

    # 1. ANOVA eta^2 for Credits_per_Course
    print("[1/7] ANOVA eta^2 (Credits_per_Course by College) ...")
    theta, boot, jack = _anova_eta_sq_task(uni_df, "Credits_per_Course")
    lo, hi, diag = _bca_ci_from_samples(boot, jack, theta)
    out["anova_credits_per_course_eta_sq"] = {
        "point": float(theta),
        "ci_low": float(lo), "ci_high": float(hi),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "stratification": "College (medicine/dentistry/Korean medicine)",
        "n_universities": int(len(uni_df)),
        "diagnostics": diag,
    }
    print(f"       point={theta:.4f}  BCa 95% CI [{lo:.4f}, {hi:.4f}]  "
          f"method={diag.get('method')}")

    # 2. ANOVA eta^2 for Total_Credits
    print("[2/7] ANOVA eta^2 (Total_Credits by College) ...")
    theta, boot, jack = _anova_eta_sq_task(uni_df, "Total_Credits")
    lo, hi, diag = _bca_ci_from_samples(boot, jack, theta)
    out["anova_total_credits_eta_sq"] = {
        "point": float(theta),
        "ci_low": float(lo), "ci_high": float(hi),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "stratification": "College (medicine/dentistry/Korean medicine)",
        "n_universities": int(len(uni_df)),
        "diagnostics": diag,
    }
    print(f"       point={theta:.4f}  BCa 95% CI [{lo:.4f}, {hi:.4f}]  "
          f"method={diag.get('method')}")

    # 3. Cohen's d for Total_Credits (Public vs Private)
    print("[3/7] Cohen's d (Total_Credits, Public vs Private) ...")
    theta, boot, jack = _cohens_d_task(uni_df, "Total_Credits")
    lo, hi, diag = _bca_ci_from_samples(boot, jack, theta)
    out["ttest_total_credits_d"] = {
        "point": float(theta),
        "ci_low": float(lo), "ci_high": float(hi),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "stratification": "Is_Public (public n=17 / private n=43)",
        "n_universities": int(len(uni_df)),
        "diagnostics": diag,
    }
    print(f"       point={theta:.4f}  BCa 95% CI [{lo:.4f}, {hi:.4f}]  "
          f"method={diag.get('method')}")

    # 4. Cramer's V for Domain x Mandatory (chi-square)
    print("[4/7] Cramer's V (Domain x Mandatory, course-level) ...")
    theta, boot, jack = _cramers_v_domain_mandatory_task(course_df)
    lo, hi, diag = _bca_ci_from_samples(boot, jack, theta)
    out["chi2_domain_mandatory_v"] = {
        "point": float(theta),
        "ci_low": float(lo), "ci_high": float(hi),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "stratification": "College (enrolment-level)",
        "n_enrollments": int(len(course_df)),
        "diagnostics": diag,
    }
    print(f"       point={theta:.4f}  BCa 95% CI [{lo:.4f}, {hi:.4f}]  "
          f"method={diag.get('method')}")

    # 5/6/7. Rank-biserial r (Adv vs FO)
    mw_tasks = [
        ("mannwhitney_advfo_credits_r", "Total_Credits",
         "rank-biserial r for Total_Credits (Adv vs FO)"),
        ("mannwhitney_advfo_breadth_r", "Breadth",
         "rank-biserial r for Breadth (Adv vs FO) -- degenerate at r=+-1.0"),
        ("mannwhitney_advfo_mandratio_r", "Mandatory_Ratio",
         "rank-biserial r for Mandatory_Ratio (Adv vs FO)"),
    ]
    for (step, (key, var, label)) in enumerate(mw_tasks, start=5):
        print(f"[{step}/7] {label} ...")
        theta, boot, jack, info = _rank_biserial_task(uni_df, var)
        lo, hi, diag = _bca_ci_from_samples(boot, jack, theta)
        out[key] = {
            "point": float(theta),
            "point_abs": float(abs(theta)),
            "ci_low": float(lo), "ci_high": float(hi),
            "ci_low_abs": float(min(abs(lo), abs(hi))),
            "ci_high_abs": float(max(abs(lo), abs(hi))),
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "stratification": "Maturity_Level in {Advanced (3), FoundationalOnly (1)}",
            "n_adv": info["n_adv"],
            "n_fo": info["n_fo"],
            "diagnostics": diag,
        }
        print(f"       point={theta:.4f}  BCa 95% CI [{lo:.4f}, {hi:.4f}]  "
              f"method={diag.get('method')}")

    meta["elapsed_seconds"] = round(time.time() - t0, 2)
    result = {"metadata": meta, "effect_sizes": out}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print()
    print(f"[OK] Wrote {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print(f"[TIMING] total elapsed: {meta['elapsed_seconds']}s")

    return result


def main() -> int:
    run_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
