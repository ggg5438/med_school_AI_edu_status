# Five-strategy classification sensitivity statistics.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config

ROOT = _SCRIPTS_DIR.parent
FINAL = config.CLASSIFICATION_DIR / "final_adjudicated_classification.csv"
M1_RULES = config.CLASSIFICATION_DIR / "m1_rules.csv"
M2_LLM = config.CLASSIFICATION_DIR / "m2_llm.csv"
UNI_META = config.UNIVERSITY_FILE
OUT_DIR = config.STATISTICS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["D1", "D2", "D3", "D4", "D5"]
DOMAIN_LABELS = {
    "D1": "Quantitative Foundations",
    "D2": "AI and Machine Learning",
    "D3": "Data Science",
    "D4": "Health Informatics",
    "D5": "Clinical AI Application",
}
PRIORITY_ORDER = ["D5", "D2", "D3", "D4", "D1"]


def load_final() -> pd.DataFrame:
    df = pd.read_csv(FINAL)
    df = df.rename(columns={"D1": "D1", "D2": "D2", "D3": "D3", "D4": "D4", "D5": "D5"})
    return df


def load_m1_rules() -> pd.DataFrame:
    df = pd.read_csv(M1_RULES)
    return df.rename(columns={
        "D1_Quantitative_Foundations": "D1",
        "D2_AI_ML": "D2",
        "D3_Data_Science": "D3",
        "D4_Health_Informatics": "D4",
        "D5_Clinical_AI_Application": "D5",
    })


def load_m2_llm(base: pd.DataFrame) -> pd.DataFrame:
    llm = pd.read_csv(M2_LLM)
    keep = ["Course_ID", "D1", "D2", "D3", "D4", "D5"]
    merged = base[["Course_ID", "Course_Name", "University", "College", "Credits", "Is_Mandatory"]].merge(
        llm[keep], on="Course_ID", how="left"
    )
    for d in DOMAINS:
        merged[d] = merged[d].fillna(0).astype(int)
    merged["n_domains"] = merged[DOMAINS].sum(axis=1)
    return merged


def load_uni_meta() -> pd.DataFrame:
    uni = pd.read_excel(UNI_META)
    uni.columns = ["College", "University", "Region", "Public_Private", "Admission_Quota"]
    region_map = {
        "서울경기인천": "Capital",
        "강원권": "NonCapital",
        "충청권": "NonCapital",
        "전라제주권": "NonCapital",
        "경남권": "NonCapital",
        "경북권": "NonCapital",
    }
    pp_map = {"국립": "Public", "공립": "Public", "사립": "Private"}
    uni["Region_Binary"] = uni["Region"].map(region_map)
    uni["Governance"] = uni["Public_Private"].map(pp_map)
    rename = {"차의과학대학교": "차의과대학교", "단국대학교 글로컬캠퍼스": "단국대학교"}
    uni["University"] = uni["University"].replace(rename)
    return uni


def apply_priority(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for d in DOMAINS:
        out[d] = 0
    for i, row in df.iterrows():
        for d in PRIORITY_ORDER:
            if df.at[i, d] == 1:
                out.at[i, d] = 1
                break
    out["n_domains"] = out[DOMAINS].sum(axis=1)
    return out


def apply_strict(df: pd.DataFrame) -> pd.DataFrame:
    n = df[DOMAINS].sum(axis=1)
    return df[n <= 1].copy()


def credit_share(df: pd.DataFrame) -> dict:
    total_unique = float(df["Credits"].sum())
    result = {}
    for d in DOMAINS:
        sub = df[df[d] == 1]
        result[d] = float(sub["Credits"].sum()) / total_unique * 100 if total_unique > 0 else 0.0
    return result


def adoption_rate(df: pd.DataFrame, all_school_keys: list[tuple]) -> dict:
    n_total = len(all_school_keys)
    school_keys_in_data = df[["University", "College"]].drop_duplicates()
    school_keys_in_data = list(zip(school_keys_in_data["University"], school_keys_in_data["College"]))
    result = {}
    for d in DOMAINS:
        sub = df[df[d] == 1][["University", "College"]].drop_duplicates()
        keys_with = set(zip(sub["University"], sub["College"]))
        n_with = sum(1 for k in all_school_keys if k in keys_with)
        result[d] = n_with / n_total * 100
    return result


def maturity_stage(df: pd.DataFrame, uni_meta: pd.DataFrame) -> pd.DataFrame:
    keys_df = uni_meta[["University", "College"]].drop_duplicates()
    if len(df) == 0:
        agg = keys_df.copy()
        for d in DOMAINS:
            agg[f"Has_{d}"] = 0
        agg["Total_Credits"] = 0.0
        agg["N_Courses"] = 0
    else:
        agg = df.groupby(["University", "College"]).agg(
            Total_Credits=("Credits", "sum"),
            N_Courses=("Credits", "count"),
        ).reset_index()
        for d in DOMAINS:
            has_d = df.groupby(["University", "College"])[d].max().rename(f"Has_{d}")
            agg = agg.merge(has_d, on=["University", "College"], how="left")
    agg = keys_df.merge(agg, on=["University", "College"], how="left")
    for c in agg.columns:
        if c.startswith("Has_") or c in ("Total_Credits", "N_Courses"):
            agg[c] = agg[c].fillna(0)
    agg["AI_Core_Count"] = agg["Has_D2"] + agg["Has_D3"] + agg["Has_D5"]

    def _stage(row):
        if row["Has_D1"] >= 1 and row["AI_Core_Count"] >= 2 and row["Total_Credits"] >= 8:
            return "Advanced"
        if row["AI_Core_Count"] >= 1:
            return "Intermediate"
        return "Foundational"
    agg["stage"] = agg.apply(_stage, axis=1)
    return agg.rename(columns={"AI_Core_Count": "n_ai_core",
                                "Total_Credits": "total_credits",
                                "Has_D1": "has_foundational"})


def fisher_or(table_2x2):
    a, b = table_2x2[0]
    c, d = table_2x2[1]
    res = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    or_sample = res.statistic
    p = res.pvalue
    try:
        from scipy.stats.contingency import odds_ratio
        or_obj = odds_ratio([[a, b], [c, d]], kind="conditional")
        lo, hi = or_obj.confidence_interval(0.95)
    except Exception:
        lo, hi = (np.nan, np.nan)
    return or_sample, lo, hi, p


def compute_strategy(name: str, df: pd.DataFrame, uni_meta: pd.DataFrame) -> dict:
    df_m = df.copy()
    all_school_keys = list(zip(uni_meta["University"], uni_meta["College"]))

    cs = credit_share(df_m)
    adopt = adoption_rate(df_m, all_school_keys)
    stage_df = maturity_stage(df_m, uni_meta)
    stage_df = stage_df.merge(uni_meta[["University", "College", "Governance", "Region_Binary"]],
                              on=["University", "College"], how="left")

    stage_counts = stage_df["stage"].value_counts().to_dict()
    for s in ["Foundational", "Intermediate", "Advanced"]:
        stage_counts.setdefault(s, 0)

    stage_df["is_adv"] = (stage_df["stage"] == "Advanced").astype(int)

    gov_pub = stage_df[stage_df["Governance"] == "Public"]
    gov_pri = stage_df[stage_df["Governance"] == "Private"]
    gov_table = [[int(gov_pub["is_adv"].sum()), int(len(gov_pub) - gov_pub["is_adv"].sum())],
                 [int(gov_pri["is_adv"].sum()), int(len(gov_pri) - gov_pri["is_adv"].sum())]]
    or_gov, gov_lo, gov_hi, p_gov = fisher_or(gov_table)

    reg_cap = stage_df[stage_df["Region_Binary"] == "Capital"]
    reg_non = stage_df[stage_df["Region_Binary"] == "NonCapital"]
    reg_table = [[int(reg_cap["is_adv"].sum()), int(len(reg_cap) - reg_cap["is_adv"].sum())],
                 [int(reg_non["is_adv"].sum()), int(len(reg_non) - reg_non["is_adv"].sum())]]
    or_reg, reg_lo, reg_hi, p_reg = fisher_or(reg_table)

    return {
        "strategy": name,
        "credit_share": cs,
        "adoption_rate": adopt,
        "stage_counts": {s: stage_counts[s] for s in ["Foundational", "Intermediate", "Advanced"]},
        "governance_or": {"or": or_gov, "ci_lo": gov_lo, "ci_hi": gov_hi, "p": p_gov,
                          "table": gov_table},
        "region_or": {"or": or_reg, "ci_lo": reg_lo, "ci_hi": reg_hi, "p": p_reg,
                      "table": reg_table},
    }


def bootstrap_domain_adoption(df: pd.DataFrame, uni_meta: pd.DataFrame,
                               n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    df_m = df.copy()
    all_school_keys = list(zip(uni_meta["University"], uni_meta["College"]))
    by_college: dict[str, list[tuple]] = {}
    for u, c in all_school_keys:
        by_college.setdefault(c, []).append((u, c))

    schools_with_d: dict[str, set] = {}
    for d in DOMAINS:
        sub = df_m[df_m[d] == 1][["University", "College"]].drop_duplicates()
        schools_with_d[d] = set(zip(sub["University"], sub["College"]))

    domain_adoptions = {d: [] for d in DOMAINS}
    n_total = len(all_school_keys)
    for _ in range(n_boot):
        resampled = []
        for c, keys in by_college.items():
            idx = rng.integers(0, len(keys), size=len(keys))
            resampled.extend([keys[i] for i in idx])
        for d in DOMAINS:
            n_with = sum(1 for k in resampled if k in schools_with_d[d])
            domain_adoptions[d].append(n_with / n_total * 100)

    observed = adoption_rate(df_m, all_school_keys)
    jack_vals = {d: [] for d in DOMAINS}
    for excl_key in all_school_keys:
        kept = [k for k in all_school_keys if k != excl_key]
        for d in DOMAINS:
            n_with = sum(1 for k in kept if k in schools_with_d[d])
            jack_vals[d].append(n_with / len(kept) * 100)

    result = {}
    z_alpha = 1.959963984540054
    for d in DOMAINS:
        boot = np.array(domain_adoptions[d])
        theta_hat = observed[d]
        z0_count = float((boot < theta_hat).sum())
        z0_total = float(len(boot))
        if z0_count == 0 or z0_count == z0_total:
            z0 = 0.0
        else:
            from scipy.stats import norm
            z0 = norm.ppf(z0_count / z0_total)
        jack = np.array(jack_vals[d])
        jack_mean = jack.mean()
        num = np.sum((jack_mean - jack) ** 3)
        den = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
        a = float(num / den) if den != 0 else 0.0
        from scipy.stats import norm
        alpha1 = norm.cdf(z0 + (z0 - z_alpha) / (1 - a * (z0 - z_alpha)))
        alpha2 = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        lo = float(np.quantile(boot, alpha1))
        hi = float(np.quantile(boot, alpha2))
        result[d] = {"observed": theta_hat, "ci_lo": lo, "ci_hi": hi,
                     "boot_mean": float(boot.mean()), "boot_sd": float(boot.std(ddof=1))}
    return result


def main():
    print("Loading data...")
    final = load_final()
    m1 = load_m1_rules()
    m2 = load_m2_llm(final)
    uni_full = load_uni_meta()
    active_keys = final[["University", "College"]].drop_duplicates()
    uni = uni_full.merge(active_keys, on=["University", "College"], how="inner")
    print(f"  Final rows: {len(final)} | m1 rows: {len(m1)} | m2 rows: {len(m2)}")
    print(f"  Universities in meta (full): {len(uni_full)} (incl. accredited but excluded)")
    print(f"  Active (Uni, College) keys: {len(uni)} (must be 60)")
    assert len(uni) == 60, f"Active sample mismatch: got {len(uni)}"

    strategies = {
        "S0_Final_Adjudicated": final.copy(),
        "S1_Single_Priority": apply_priority(final),
        "S2_Strict_Match": apply_strict(final),
        "S3_LLM_Only": m2.copy(),
        "S4_Rule_Only": m1.copy(),
    }
    print("\nStrategies:")
    for name, df in strategies.items():
        n_courses = len(df)
        ncomp = int((df[DOMAINS].sum(axis=1) > 1).sum())
        print(f"  {name}: {n_courses} courses, {ncomp} multi-domain")

    print("\nComputing per-strategy metrics...")
    results = {}
    for name, df in strategies.items():
        results[name] = compute_strategy(name, df, uni)

    rows = []
    for name, r in results.items():
        for d in DOMAINS:
            rows.append({
                "strategy": name,
                "domain": d,
                "domain_label": DOMAIN_LABELS[d],
                "credit_share_pct": round(r["credit_share"][d], 2),
                "adoption_rate_pct": round(r["adoption_rate"][d], 2),
            })
        rows.append({
            "strategy": name, "domain": "STAGE", "domain_label": "Foundational",
            "credit_share_pct": r["stage_counts"]["Foundational"],
            "adoption_rate_pct": None,
        })
        rows.append({
            "strategy": name, "domain": "STAGE", "domain_label": "Intermediate",
            "credit_share_pct": r["stage_counts"]["Intermediate"],
            "adoption_rate_pct": None,
        })
        rows.append({
            "strategy": name, "domain": "STAGE", "domain_label": "Advanced",
            "credit_share_pct": r["stage_counts"]["Advanced"],
            "adoption_rate_pct": None,
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "sensitivity_full.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'sensitivity_full.csv'}")

    summary = {}
    for name, r in results.items():
        summary[name] = {
            "credit_share_pct": {d: round(r["credit_share"][d], 2) for d in DOMAINS},
            "adoption_rate_pct": {d: round(r["adoption_rate"][d], 2) for d in DOMAINS},
            "stage_counts": r["stage_counts"],
            "governance_or": {
                "or_sample": round(r["governance_or"]["or"], 3),
                "ci_lo": round(r["governance_or"]["ci_lo"], 3),
                "ci_hi": round(r["governance_or"]["ci_hi"], 3),
                "p_value": round(r["governance_or"]["p"], 4),
                "contingency_PubPri_AdvNot": r["governance_or"]["table"],
            },
            "region_or": {
                "or_sample": round(r["region_or"]["or"], 3),
                "ci_lo": round(r["region_or"]["ci_lo"], 3),
                "ci_hi": round(r["region_or"]["ci_hi"], 3),
                "p_value": round(r["region_or"]["p"], 4),
                "contingency_CapNonCap_AdvNot": r["region_or"]["table"],
            },
        }
    with open(OUT_DIR / "sensitivity_full_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {OUT_DIR / 'sensitivity_full_summary.json'}")

    print("\nBootstrapping domain adoption rate (1,000 College-stratified resamples, seed 42)...")
    bca = bootstrap_domain_adoption(final, uni, n_boot=1000, seed=42)
    bca_rows = []
    for d in DOMAINS:
        bca_rows.append({
            "domain": d, "domain_label": DOMAIN_LABELS[d],
            "observed_pct": round(bca[d]["observed"], 2),
            "bca_ci_lo": round(bca[d]["ci_lo"], 2),
            "bca_ci_hi": round(bca[d]["ci_hi"], 2),
            "boot_mean": round(bca[d]["boot_mean"], 2),
            "boot_sd": round(bca[d]["boot_sd"], 2),
        })
    pd.DataFrame(bca_rows).to_csv(OUT_DIR / "bootstrap_domain_adoption_bca.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'bootstrap_domain_adoption_bca.csv'}")

    print("\nHeadline (Advanced n by strategy):")
    for name, r in results.items():
        print(f"  {name}: n_Adv = {r['stage_counts']['Advanced']}, "
              f"GovOR = {r['governance_or']['or']:.2f} "
              f"(95% CI {r['governance_or']['ci_lo']:.2f}-{r['governance_or']['ci_hi']:.2f}; "
              f"P={r['governance_or']['p']:.3f}); "
              f"RegOR = {r['region_or']['or']:.2f} "
              f"(95% CI {r['region_or']['ci_lo']:.2f}-{r['region_or']['ci_hi']:.2f}; "
              f"P={r['region_or']['p']:.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
