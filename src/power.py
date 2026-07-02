# -*- coding: utf-8 -*-
"""
Retrospective power / minimum detectable effect (MDE) for the fixed national
cohort (n=63; 40 Medicine / 11 Dentistry / 12 Korean Medicine; 17 public /
46 private).

MDE at 80% power, alpha = 0.05 (two-sided). Monte Carlo tests use n_sim=2000
with seed 42; analytic tests use statsmodels closed forms.

  1) Fisher exact: public (17) vs private (46) x reference-configuration vs other
  2) Kruskal-Wallis: total credits across {Medicine=40, Dentistry=11, KM=12}
  3) One-way ANOVA: total credits across 3 college types
  4) Independent t-test: total credits, public (17) vs private (46)

Reads results/statistics/per_school.csv.
Output: results/statistics/power_analysis.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower

ALPHA = 0.05
TARGET_POWER = 0.80
N_SIM = 2000
RNG_SEED = 42

_SRC = Path(__file__).resolve().parent
IN_CSV = _SRC.parent / "results" / "statistics" / "per_school.csv"
OUT_DIR = _SRC.parent / "results" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bisect_mde(effect_power_fn, lo, hi, tol=1e-3, max_iter=80):
    p_lo = effect_power_fn(lo)
    p_hi = effect_power_fn(hi)
    if p_hi < TARGET_POWER:
        raise ValueError(f"Upper bound hi={hi} gives power {p_hi:.3f} < {TARGET_POWER}.")
    if p_lo >= TARGET_POWER:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = effect_power_fn(mid)
        if p_mid >= TARGET_POWER:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return hi


def fisher_mde_or(n_public, n_private, p_base, alpha=ALPHA, power=TARGET_POWER,
                  n_sim=N_SIM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    p0 = p_base

    def power_at(or_value):
        p1 = (or_value * p0) / (1.0 - p0 + or_value * p0)
        x_pub = rng.binomial(n_public, p1, size=n_sim)
        x_priv = rng.binomial(n_private, p0, size=n_sim)
        rej = 0
        for a, c in zip(x_pub, x_priv):
            b = n_public - a
            d = n_private - c
            _, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
            if p < alpha:
                rej += 1
        return rej / n_sim

    try:
        or_upper = _bisect_mde(power_at, lo=1.0, hi=50.0, tol=0.05, max_iter=40)
        or_upper_feasible = True
    except ValueError:
        or_upper = float("inf")
        or_upper_feasible = False

    def power_at_inv(inv_or):
        return power_at(1.0 / inv_or)

    try:
        or_lower_inv = _bisect_mde(power_at_inv, lo=1.0, hi=50.0, tol=0.05, max_iter=40)
        or_lower = 1.0 / or_lower_inv
        or_lower_feasible = True
        max_power_at_zero = None
    except ValueError:
        or_lower = float("nan")
        or_lower_feasible = False
        max_power_at_zero = round(power_at(0.01), 3)

    return {
        "n_public": n_public,
        "n_private": n_private,
        "baseline_refconfig_prob_private": round(p0, 4),
        "alpha": alpha,
        "target_power": power,
        "n_sim": n_sim,
        "mde_or_upper": round(or_upper, 2) if or_upper_feasible else None,
        "mde_or_upper_feasible": or_upper_feasible,
        "mde_or_lower": round(or_lower, 2) if or_lower_feasible else None,
        "mde_or_lower_feasible": or_lower_feasible,
        "max_power_at_lower_extreme": max_power_at_zero,
        "method": "Monte Carlo simulation of Fisher exact, two-sided",
    }


def kruskal_mde_epsilon_sq(group_ns, alpha=ALPHA, power=TARGET_POWER,
                           n_sim=N_SIM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    k = len(group_ns)
    n_total = sum(group_ns)

    def empirical_eps_sq(delta, n_per_group=10_000):
        rng_ref = np.random.default_rng(0)
        means = [0.0, delta / 2, delta]
        samples = [rng_ref.normal(m, 1.0, size=n_per_group) for m in means]
        H, _ = stats.kruskal(*samples)
        n = k * n_per_group
        return (H - k + 1) / (n - k)

    def power_at(delta):
        means = [0.0, delta / 2, delta]
        rej = 0
        for _ in range(n_sim):
            samples = [rng.normal(m, 1.0, size=n) for m, n in zip(means, group_ns)]
            H, p = stats.kruskal(*samples)
            if p < alpha:
                rej += 1
        return rej / n_sim

    delta_mde = _bisect_mde(power_at, lo=0.0, hi=3.0, tol=0.01)
    eps_sq_mde = empirical_eps_sq(delta_mde)
    return {
        "group_ns": group_ns,
        "n_total": n_total,
        "alpha": alpha,
        "target_power": power,
        "n_sim": n_sim,
        "mde_delta_sd_units": round(delta_mde, 3),
        "mde_epsilon_sq": round(eps_sq_mde, 3),
        "method": "Monte Carlo: 3 Normal groups, equally spaced means, sd=1",
    }


def anova_mde_f(group_ns, alpha=ALPHA, power=TARGET_POWER):
    n_total = sum(group_ns)
    k = len(group_ns)
    f_mde = FTestAnovaPower().solve_power(
        effect_size=None, nobs=n_total, alpha=alpha, power=power, k_groups=k)
    eta_sq = f_mde ** 2 / (1 + f_mde ** 2)
    return {
        "group_ns": group_ns,
        "n_total": n_total,
        "k_groups": k,
        "alpha": alpha,
        "target_power": power,
        "mde_cohens_f": round(float(f_mde), 3),
        "mde_eta_sq": round(float(eta_sq), 3),
        "method": "statsmodels FTestAnovaPower (analytic)",
        "caveat": "Analytic formula assumes equal group sizes; with unequal "
                  f"sizes {group_ns}, the true MDE is slightly larger.",
    }


def ttest_mde_d(n1, n2, alpha=ALPHA, power=TARGET_POWER):
    d_mde = TTestIndPower().solve_power(
        effect_size=None, nobs1=n1, alpha=alpha, power=power,
        ratio=n2 / n1, alternative="two-sided")
    return {
        "n1_public": n1,
        "n2_private": n2,
        "alpha": alpha,
        "target_power": power,
        "mde_cohens_d": round(float(d_mde), 3),
        "method": "statsmodels TTestIndPower (analytic)",
    }


def main():
    df = pd.read_csv(IN_CSV)
    n_med = int((df["College"] == "Medicine").sum())
    n_dent = int((df["College"] == "Dentistry").sum())
    n_km = int((df["College"] == "Korean Medicine").sum())
    is_public = df["Public_Private"] == "국립"
    n_public = int(is_public.sum())
    n_private = int((~is_public).sum())
    ref = df["meets_reference_config"] == 1
    p_ref_priv = float(ref[~is_public].mean())

    print(f"Cohort n={len(df)} (Med={n_med}, Dent={n_dent}, KM={n_km}; "
          f"public={n_public}, private={n_private})")
    print(f"Reference-config baseline in private arm: {p_ref_priv:.4f}")

    bundle = {
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "cohort": {
            "n_total": int(len(df)),
            "medicine": n_med,
            "dentistry": n_dent,
            "korean_medicine": n_km,
            "public": n_public,
            "private": n_private,
            "observed_p_refconfig_private": round(p_ref_priv, 4),
        },
        "tests": {
            "fisher_exact_public_private_refconfig":
                fisher_mde_or(n_public, n_private, p_ref_priv),
            "kruskal_total_credits_by_college":
                kruskal_mde_epsilon_sq([n_med, n_dent, n_km]),
            "anova_total_credits_by_college":
                anova_mde_f([n_med, n_dent, n_km]),
            "t_test_total_credits_public_vs_private":
                ttest_mde_d(n_public, n_private),
        },
        "rng_seed": RNG_SEED,
        "n_sim_non_analytic": N_SIM,
        "input": "results/statistics/per_school.csv",
    }

    out = OUT_DIR / "power_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {out}")
    for name, t in bundle["tests"].items():
        print(name, {k: v for k, v in t.items()
                     if k.startswith("mde") or k == "max_power_at_lower_extreme"})


if __name__ == "__main__":
    main()
