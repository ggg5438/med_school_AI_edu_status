# Descriptive and inferential analysis pipeline.
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config
import analysis
from data_loader import build_university_features, compute_maturity, MATURITY_LABELS
from classification_loader import (
    load_all, load_course_df, load_university_info,
    reclassify_with_strategy,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Paths
PROJECT_ROOT = _SCRIPTS_DIR.parent
STATS_DIR = config.STATISTICS_DIR
SENS_DIR = PROJECT_ROOT / "results" / "sensitivity"
CONSENSUS_CSV = config.CLASSIFICATION_DIR / "consensus.csv"


def ensure_dirs() -> None:
    for d in (STATS_DIR, SENS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Save helpers
def save_results(results: dict) -> None:
    out = STATS_DIR

    desc = results["descriptive"]
    desc["table1"].to_csv(out / "table1_descriptive.csv", index=False,
                          encoding="utf-8-sig")
    desc["domain_distribution"].to_csv(out / "domain_distribution.csv",
                                        index=False, encoding="utf-8-sig")

    results["anova"]["anova_main"].to_csv(out / "anova_main.csv", index=False)
    results["anova"]["anova_domains"].to_csv(out / "anova_domains.csv",
                                              index=False)

    results["public_private"]["pp_tests"].to_csv(
        out / "public_private.csv", index=False
    )

    results["temporal"]["temporal_df"].to_csv(
        out / "temporal_placement.csv", index=False, encoding="utf-8-sig"
    )
    results["temporal"]["contingency"].to_csv(
        out / "temporal_contingency.csv", encoding="utf-8-sig"
    )

    results["mandatory"]["domain_mandatory"].to_csv(
        out / "mandatory_by_domain.csv", index=False, encoding="utf-8-sig"
    )
    results["mandatory"]["college_mandatory"].to_csv(
        out / "mandatory_by_college.csv", index=False, encoding="utf-8-sig"
    )

    results["maturity"]["college_cross"].to_csv(
        out / "maturity_by_college.csv", encoding="utf-8-sig"
    )
    results["maturity"]["pp_cross"].to_csv(
        out / "maturity_by_pp.csv", encoding="utf-8-sig"
    )

    mp = results["maturity_predictors"]
    if isinstance(mp.get("ordinal_logistic"), pd.DataFrame) and len(mp["ordinal_logistic"]) > 0:
        mp["ordinal_logistic"].to_csv(
            out / "maturity_predictors.csv", index=False
        )
    with open(out / "maturity_predictors_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(mp.get("ordinal_summary", {}), f, indent=2,
                  ensure_ascii=False)

    fe = mp.get("fisher_exact", {})
    payload = {k: v for k, v in fe.items() if k != "contingency"}
    out_payload = {
        "odds_ratio": payload.get("odds_ratio"),
        "odds_ratio_sample": payload.get("odds_ratio_sample"),
        "odds_ratio_conditional_mle": payload.get("odds_ratio_conditional_mle"),
        "odds_ratio_reference": payload.get("odds_ratio_reference"),
        "ci_95_lower": payload.get("odds_ratio_ci_low"),
        "ci_95_upper": payload.get("odds_ratio_ci_high"),
        "p_value": payload.get("p_value"),
        "method": ("Fisher exact (scipy.stats.fisher_exact) for P-value; "
                   "exact 95% CI from conditional MLE "
                   "(scipy.stats.contingency.odds_ratio, kind='conditional')."),
        "confidence_level": 0.95,
    }
    ct = fe.get("contingency")
    if ct is not None and hasattr(ct, "to_dict"):
        out_payload["contingency_table"] = {
            str(idx): {str(c): int(v) for c, v in row.items()}
            for idx, row in ct.to_dict(orient="index").items()
        }
    with open(out / "fisher_exact.json", "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    results["maturity_comparison"].to_csv(
        out / "maturity_comparison.csv", index=False, encoding="utf-8-sig"
    )

    results["gap_analysis"]["gap_df"].to_csv(
        out / "gap_analysis.csv", index=False, encoding="utf-8-sig"
    )
    results["gap_analysis"]["summary"].to_csv(
        out / "gap_summary.csv", encoding="utf-8-sig"
    )

    results["effect_sizes"].to_csv(
        out / "effect_sizes.csv", index=False, encoding="utf-8-sig"
    )


# Sensitivity analyses (5 strategies + College-stratified bootstrap)
def _fisher_from_counts(pub_adv, pub_other, priv_adv, priv_other) -> dict:
    from scipy import stats as _stats
    ct = np.array([[pub_adv, pub_other], [priv_adv, priv_other]])
    OR, p = _stats.fisher_exact(ct, alternative="two-sided")
    try:
        from scipy.stats.contingency import odds_ratio as _spor
        r = _spor(ct, kind="conditional")
        ci = r.confidence_interval(confidence_level=0.95)
        return {
            "odds_ratio": float(OR),
            "odds_ratio_cond_mle": float(r.statistic),
            "ci_low": float(ci.low),
            "ci_high": float(ci.high),
            "p_value": float(p),
            "contingency_2x2": ct.tolist(),
        }
    except Exception as e:
        return {
            "odds_ratio": float(OR),
            "ci_method_error": str(e),
            "p_value": float(p),
            "contingency_2x2": ct.tolist(),
        }


def _compute_adv_stats(uni_df: pd.DataFrame) -> dict:
    dist = uni_df["Maturity_Label"].value_counts().to_dict()
    n_advanced = int(dist.get("Advanced", 0))
    n_intermediate = int(dist.get("Intermediate", 0))
    n_foundational = int(dist.get("Foundational-Only", 0))
    n_minimal = int(dist.get("Minimal", 0))

    is_pub = uni_df["Is_Public"] == 1
    is_adv = uni_df["Maturity_Level"] == 3
    pub_adv = int((is_pub & is_adv).sum())
    pub_other = int((is_pub & ~is_adv).sum())
    priv_adv = int((~is_pub & is_adv).sum())
    priv_other = int((~is_pub & ~is_adv).sum())
    fisher = _fisher_from_counts(pub_adv, pub_other, priv_adv, priv_other)

    return {
        "maturity_distribution": {
            "Advanced": n_advanced,
            "Intermediate": n_intermediate,
            "Foundational-Only": n_foundational,
            "Minimal": n_minimal,
        },
        "contingency": {
            "pub_adv": pub_adv, "pub_other": pub_other,
            "priv_adv": priv_adv, "priv_other": priv_other,
        },
        "fisher_odds_ratio": fisher.get("odds_ratio"),
        "fisher_ci_low": fisher.get("ci_low"),
        "fisher_ci_high": fisher.get("ci_high"),
        "fisher_p": fisher.get("p_value"),
    }


def _strategy_to_uni_df(strategy: str, uni_info: pd.DataFrame,
                        consensus_course_df: pd.DataFrame) -> pd.DataFrame:
    if strategy == "S1_consensus":
        course_df = consensus_course_df
    elif strategy == "S2_single_domain_priority":
        course_df = reclassify_with_strategy(consensus_course_df,
                                             "single_domain_priority")
    elif strategy == "S3_strict_match":
        course_df = reclassify_with_strategy(consensus_course_df,
                                             "strict_match")
    elif strategy == "S4_llm_only":
        course_df = load_course_df(classifier="m2")
    elif strategy == "S5_rule_only":
        course_df = load_course_df(classifier="m1")
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    uni_df = build_university_features(course_df, uni_info)
    uni_df["Maturity_Level"] = compute_maturity(uni_df)
    uni_df["Maturity_Label"] = uni_df["Maturity_Level"].map(MATURITY_LABELS)
    return uni_df


def run_sensitivity(consensus_course_df: pd.DataFrame,
                    uni_info: pd.DataFrame,
                    uni_df: pd.DataFrame) -> dict:
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    rng_seed = config.RANDOM_STATE
    n_boot = 1000

    strategies = [
        "S1_consensus",
        "S2_single_domain_priority",
        "S3_strict_match",
        "S4_llm_only",
        "S5_rule_only",
    ]
    strategy_out = {}
    for s in strategies:
        u = _strategy_to_uni_df(s, uni_info, consensus_course_df)
        stats_ = _compute_adv_stats(u)
        stats_["total_credits_sum"] = float(u["Total_Credits"].sum())
        stats_["total_credits_mean"] = float(u["Total_Credits"].mean())
        stats_["pct_schools_any_ai_core"] = float((u["Has_AI_Core"] == 1).mean() * 100)
        stats_["per_domain_school_pct"] = {
            d: float(u[f"Has_{d}"].mean() * 100)
            for d in config.DOMAIN_COLS
        }
        strategy_out[s] = stats_
        print(f"  [{s}] Advanced={stats_['maturity_distribution']['Advanced']} "
              f"OR={stats_.get('fisher_odds_ratio'):.3f} "
              f"p={stats_.get('fisher_p'):.4f}")

    rng = np.random.default_rng(rng_seed)
    college_groups = list(uni_df.groupby("College").groups.keys())
    group_indices = {c: np.array(uni_df.index[uni_df["College"] == c])
                     for c in college_groups}

    boot_rows = {
        "total_credits_mean": [],
        "advanced_count": [],
        "intermediate_count": [],
        "foundational_only_count": [],
        "or_sample": [],
        "p_value": [],
    }
    for b in range(n_boot):
        resampled_idx = np.concatenate([
            rng.choice(idxs, size=len(idxs), replace=True)
            for idxs in group_indices.values()
        ])
        u_boot = uni_df.loc[resampled_idx].copy().reset_index(drop=True)

        boot_rows["total_credits_mean"].append(
            float(u_boot["Total_Credits"].mean())
        )
        dist = u_boot["Maturity_Label"].value_counts().to_dict()
        boot_rows["advanced_count"].append(int(dist.get("Advanced", 0)))
        boot_rows["intermediate_count"].append(int(dist.get("Intermediate", 0)))
        boot_rows["foundational_only_count"].append(
            int(dist.get("Foundational-Only", 0))
        )

        is_pub = u_boot["Is_Public"] == 1
        is_adv = u_boot["Maturity_Level"] == 3
        pub_adv = int((is_pub & is_adv).sum())
        pub_other = int((is_pub & ~is_adv).sum())
        priv_adv = int((~is_pub & is_adv).sum())
        priv_other = int((~is_pub & ~is_adv).sum())
        if (pub_adv + pub_other) == 0 or (priv_adv + priv_other) == 0 \
                or (pub_adv + priv_adv) == 0 or (pub_other + priv_other) == 0:
            boot_rows["or_sample"].append(float("nan"))
            boot_rows["p_value"].append(float("nan"))
        else:
            fe = _fisher_from_counts(pub_adv, pub_other, priv_adv, priv_other)
            boot_rows["or_sample"].append(fe.get("odds_ratio", float("nan")))
            boot_rows["p_value"].append(fe.get("p_value", float("nan")))

    def _ci(values):
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            return (float("nan"), float("nan"), float("nan"), 0)
        return (
            float(np.mean(arr)),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
            int(len(arr)),
        )

    s6_out = {}
    for metric, vals in boot_rows.items():
        mean, lo, hi, n = _ci(vals)
        s6_out[metric] = {
            "mean": mean,
            "ci_2_5_pct": lo,
            "ci_97_5_pct": hi,
            "n_valid_bootstraps": n,
        }

    rows = []
    for s in strategies:
        v = strategy_out[s]
        rows.append({
            "strategy": s,
            "advanced_n": v["maturity_distribution"]["Advanced"],
            "intermediate_n": v["maturity_distribution"]["Intermediate"],
            "foundational_only_n": v["maturity_distribution"]["Foundational-Only"],
            "minimal_n": v["maturity_distribution"]["Minimal"],
            "total_credits_sum": v["total_credits_sum"],
            "total_credits_mean": v["total_credits_mean"],
            "pct_schools_any_ai_core": v["pct_schools_any_ai_core"],
            "fisher_or": v.get("fisher_odds_ratio"),
            "fisher_ci_low": v.get("fisher_ci_low"),
            "fisher_ci_high": v.get("fisher_ci_high"),
            "fisher_p": v.get("fisher_p"),
            "D1_school_pct": v["per_domain_school_pct"]["D1_Quantitative_Foundations"],
            "D2_school_pct": v["per_domain_school_pct"]["D2_AI_ML"],
            "D3_school_pct": v["per_domain_school_pct"]["D3_Data_Science"],
            "D4_school_pct": v["per_domain_school_pct"]["D4_Health_Informatics"],
            "D5_school_pct": v["per_domain_school_pct"]["D5_Clinical_AI_Application"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(SENS_DIR / "sensitivity_summary.csv", index=False,
              encoding="utf-8-sig")

    rows_s6 = [{"metric": m, **s6_out[m]} for m in s6_out]
    pd.DataFrame(rows_s6).to_csv(
        SENS_DIR / "bootstrap_ci.csv", index=False, encoding="utf-8-sig"
    )

    out = {
        "strategies": strategy_out,
        "bootstrap": s6_out,
        "bootstrap_config": {
            "n_resamples": n_boot,
            "seed": rng_seed,
            "stratification": "College",
            "unit": "university",
        },
        "note": (
            "S1 is the primary strategy. S2/S3 derive from S1 by post-hoc "
            "re-weighting of the consensus matrix. S4/S5 replay the "
            "LLM-assistant-only / rule-based-only classifier outputs "
            "respectively. The bootstrap resamples universities stratified "
            "by College, 1000 replicates, seed=42."
        ),
    }
    with open(SENS_DIR / "sensitivity_all.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out


# Print helpers
def print_highlights(results: dict) -> None:
    print()
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    desc = results["descriptive"]
    print(f"Courses: {desc['n_courses']}, Universities: {desc['n_universities']}")
    print(f"Total credits: {desc['total_credits']:.0f}")
    print(f"Mandatory ratio: {desc['mandatory_ratio']:.1%}")

    print("\nDomain distribution:")
    for _, row in desc["domain_distribution"].iterrows():
        print(f"  {row['Domain']}: {row['Total_Credits']:.1f} cr "
              f"({row['Pct_Credits']:.1f}%), "
              f"{int(row['N_Schools'])} schools ({row['Pct_Schools']:.1f}%)")

    pp = results["public_private"]["pp_tests"]
    r = pp[pp["Variable"] == "Total_Credits"].iloc[0]
    print(f"\nPublic vs Private (Total Credits):")
    print(f"  Pub={r['Public_Mean']:.2f}, Priv={r['Private_Mean']:.2f}, "
          f"d={r['Cohens_d']:.2f}, p={r['p']:.4f}")

    print("\nMaturity distribution:")
    for level, count in results["maturity"]["distribution"].items():
        print(f"  {level}: {count}")

    fe = results["maturity_predictors"].get("fisher_exact", {})
    if fe.get("p_value") is not None:
        print(f"\nFisher exact (Advanced x Governance):")
        lo = fe.get("odds_ratio_ci_low")
        hi = fe.get("odds_ratio_ci_high")
        print(f"  OR={fe['odds_ratio']:.2f} (95% CI {lo:.2f}-{hi:.2f}) "
              f"p={fe['p_value']:.4f}")


# Main
def main() -> int:
    t0 = time.time()
    print("=" * 74)
    print("DESCRIPTIVE + INFERENTIAL ANALYSIS")
    print("=" * 74)

    ensure_dirs()
    if not CONSENSUS_CSV.exists():
        print(f"[FATAL] Classification CSV not found: {CONSENSUS_CSV}",
              file=sys.stderr)
        print("        See README.md for the expected data/ layout.",
              file=sys.stderr)
        return 2

    course_df, uni_info, uni_df = load_all(classifier="consensus")

    print("\n[STEP] Running analysis modules ...")
    results = analysis.run_all(course_df, uni_df)

    print("\n[STEP] Saving statistics ...")
    save_results(results)

    print_highlights(results)

    print("\n[STEP] Running sensitivity analyses (5 strategies + bootstrap) ...")
    run_sensitivity(course_df, uni_info, uni_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Results -> {PROJECT_ROOT / 'results'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
