# Advanced-threshold sensitivity sweep (5..12 credits).
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

CREDIT_THRESHOLDS = list(range(5, 13))


def maturity_at(uni_df: pd.DataFrame, total_credit_threshold: int) -> pd.Series:
    n_aicore_domains = (
        (uni_df["Has_D2_AI_ML"] == 1).astype(int)
        + (uni_df["Has_D3_Data_Science"] == 1).astype(int)
        + (uni_df["Has_D5_Clinical_AI_Application"] == 1).astype(int)
    )
    has_foundational = uni_df["Has_Foundational"] == 1
    has_ai_core = uni_df["Has_AI_Core"] == 1
    total_cr = uni_df["Total_Credits"]
    is_advanced = (
        has_ai_core & has_foundational & (n_aicore_domains >= 2)
        & (total_cr >= total_credit_threshold)
    )
    levels = pd.Series(2, index=uni_df.index)
    levels[total_cr == 0] = 0
    levels[~has_ai_core] = 1
    levels[is_advanced] = 3
    return levels


def fisher_or(a: int, b: int, c: int, d: int) -> dict:
    ct = np.array([[a, b], [c, d]])
    OR_sample, p = sstats.fisher_exact(ct, alternative="two-sided")
    try:
        r = scipy_odds_ratio(ct, kind="conditional")
        ci = r.confidence_interval(confidence_level=0.95)
        return {
            "odds_ratio_sample": float(OR_sample),
            "odds_ratio_cmle": float(r.statistic),
            "ci_95_low": float(ci.low),
            "ci_95_high": float(ci.high),
            "p_value": float(p),
        }
    except Exception:
        return {
            "odds_ratio_sample": float(OR_sample),
            "odds_ratio_cmle": float("nan"),
            "ci_95_low": float("nan"),
            "ci_95_high": float("nan"),
            "p_value": float(p),
        }


def main() -> int:
    print("=" * 70)
    print("THRESHOLD SWEEP — Advanced threshold X=5..12 credits")
    print("=" * 70)
    course_df, uni_info, uni_df = load_all(classifier="consensus")
    uni_df = uni_df.copy()
    uni_df["Region_Binary"] = np.where(
        uni_df["Region"] == "Seoul/Gyeonggi", "Seoul", "Non-Seoul"
    )

    sweep_rows = []
    fisher_rows = []
    for x in CREDIT_THRESHOLDS:
        levels = maturity_at(uni_df, x)
        n_adv = int((levels == 3).sum())
        n_int = int((levels == 2).sum())
        n_fnd = int((levels == 1).sum())
        n_min = int((levels == 0).sum())

        adv_mask = levels == 3
        adv_med = int((adv_mask & (uni_df["College"] == "Medicine")).sum())
        adv_dent = int((adv_mask & (uni_df["College"] == "Dentistry")).sum())
        adv_km = int((adv_mask & (uni_df["College"] == "Korean Medicine")).sum())
        adv_pub = int((adv_mask & (uni_df["Is_Public"] == 1)).sum())
        adv_priv = int((adv_mask & (uni_df["Is_Public"] == 0)).sum())
        adv_seoul = int((adv_mask & (uni_df["Region_Binary"] == "Seoul")).sum())
        adv_nons = int((adv_mask & (uni_df["Region_Binary"] == "Non-Seoul")).sum())

        sweep_rows.append({
            "Threshold": x,
            "N_Advanced": n_adv,
            "N_Intermediate": n_int,
            "N_Foundational_Only": n_fnd,
            "N_Minimal": n_min,
            "Pct_Advanced": float(n_adv / len(uni_df) * 100),
            "Adv_Medicine": adv_med,
            "Adv_Dentistry": adv_dent,
            "Adv_KoreanMedicine": adv_km,
            "Adv_Public": adv_pub,
            "Adv_Private": adv_priv,
            "Adv_Seoul": adv_seoul,
            "Adv_NonSeoul": adv_nons,
        })

        pub_n = int((uni_df["Is_Public"] == 1).sum())
        priv_n = int((uni_df["Is_Public"] == 0).sum())
        gov = fisher_or(
            adv_pub, pub_n - adv_pub,
            adv_priv, priv_n - adv_priv,
        )
        seoul_n = int((uni_df["Region_Binary"] == "Seoul").sum())
        nons_n = int((uni_df["Region_Binary"] == "Non-Seoul").sum())
        reg = fisher_or(
            adv_seoul, seoul_n - adv_seoul,
            adv_nons, nons_n - adv_nons,
        )
        fisher_rows.append({
            "Threshold": x,
            "Gov_OR_sample": gov["odds_ratio_sample"],
            "Gov_CI_low": gov["ci_95_low"],
            "Gov_CI_high": gov["ci_95_high"],
            "Gov_p": gov["p_value"],
            "Region_OR_sample": reg["odds_ratio_sample"],
            "Region_CI_low": reg["ci_95_low"],
            "Region_CI_high": reg["ci_95_high"],
            "Region_p": reg["p_value"],
        })

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_DIR / "threshold_sweep.csv", index=False,
                 encoding="utf-8-sig")
    fisher = pd.DataFrame(fisher_rows)
    fisher.to_csv(OUT_DIR / "threshold_sweep_fisher.csv", index=False,
                  encoding="utf-8-sig")

    print("\n[Sweep]")
    print(sweep.to_string(index=False))
    print("\n[Fisher]")
    print(fisher.to_string(index=False))

    n_uni = len(uni_df)
    summary = {
        "thresholds": CREDIT_THRESHOLDS,
        "n_universities": n_uni,
        "primary_threshold": 8,
        "robustness": {
            "Gov_OR_direction_consistent": bool(
                np.all(np.array(fisher["Gov_OR_sample"]) > 1.0)
            ),
            "Gov_CI_includes_1_at_all_thresholds": bool(
                np.all(
                    (fisher["Gov_CI_low"] <= 1.0)
                    & (fisher["Gov_CI_high"] >= 1.0)
                )
            ),
            "Region_OR_direction_consistent": bool(
                np.all(np.array(fisher["Region_OR_sample"]) > 1.0)
                or np.all(np.array(fisher["Region_OR_sample"]) < 1.0)
            ),
            "Region_CI_includes_1_at_all_thresholds": bool(
                np.all(
                    (fisher["Region_CI_low"] <= 1.0)
                    & (fisher["Region_CI_high"] >= 1.0)
                )
            ),
        },
        "advanced_count_at_each_threshold": dict(
            zip(map(int, sweep["Threshold"]), map(int, sweep["N_Advanced"]))
        ),
    }
    with open(OUT_DIR / "threshold_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {OUT_DIR / 'threshold_sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
