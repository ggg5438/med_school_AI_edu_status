# Fisher exact odds ratio with exact 95% CI: governance vs Advanced maturity.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import odds_ratio as sp_odds_ratio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FISHER_JSON = PROJECT_ROOT / "results" / "statistics" / "fisher_exact.json"
MATURITY_BY_PP = PROJECT_ROOT / "results" / "statistics" / "maturity_by_pp.csv"


def compute_fisher_with_ci(
    public_advanced: int,
    public_other: int,
    private_advanced: int,
    private_other: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    table = [
        [public_advanced, public_other],
        [private_advanced, private_other],
    ]

    or_sample, p_value = stats.fisher_exact(table, alternative="two-sided")

    res_cond = sp_odds_ratio(table, kind="conditional")
    ci = res_cond.confidence_interval(confidence_level=confidence)

    out = {
        "odds_ratio_sample": float(or_sample),
        "odds_ratio_conditional_mle": float(res_cond.statistic),
        "ci_95_lower": float(ci.low),
        "ci_95_upper": float(ci.high),
        "p_value": float(p_value),
        "method": (
            "Fisher exact (scipy.stats.fisher_exact) for P-value; "
            "exact 95% CI from conditional MLE "
            "(scipy.stats.contingency.odds_ratio, kind='conditional')."
        ),
        "odds_ratio_reference": "private",
        "contingency_table": {
            "Public": {"Advanced": public_advanced, "Other": public_other},
            "Private": {"Advanced": private_advanced, "Other": private_other},
        },
        "confidence_level": confidence,
    }
    return out


def update_fisher_json(result: dict[str, Any], path: Path = FISHER_JSON) -> None:
    payload = {
        "odds_ratio": round(result["odds_ratio_sample"], 3),
        "odds_ratio_sample": round(result["odds_ratio_sample"], 3),
        "odds_ratio_conditional_mle": round(result["odds_ratio_conditional_mle"], 3),
        "odds_ratio_reference": result["odds_ratio_reference"],
        "ci_95_lower": round(result["ci_95_lower"], 3),
        "ci_95_upper": round(result["ci_95_upper"], 3),
        "p_value": result["p_value"],
        "method": result["method"],
        "contingency_table": result["contingency_table"],
        "confidence_level": result["confidence_level"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_summary(result: dict[str, Any]) -> None:
    print("=== Fisher exact (Public vs Private, reference = Private) ===")
    ct = result["contingency_table"]
    print(
        f"  Contingency: Public Adv={ct['Public']['Advanced']} Other={ct['Public']['Other']}; "
        f"Private Adv={ct['Private']['Advanced']} Other={ct['Private']['Other']}"
    )
    print(f"  OR (sample)            = {result['odds_ratio_sample']:.3f}")
    print(f"  OR (conditional MLE)   = {result['odds_ratio_conditional_mle']:.3f}")
    print(
        f"  95% CI (exact)         = "
        f"[{result['ci_95_lower']:.3f}, {result['ci_95_upper']:.3f}]"
    )
    print(f"  P-value (Fisher exact) = {result['p_value']:.4f}")
    print(
        f"  Reported as: OR={result['odds_ratio_sample']:.2f} "
        f"(95% CI {result['ci_95_lower']:.2f}-{result['ci_95_upper']:.2f}), "
        f"P={result['p_value']:.3f}, reference: {result['odds_ratio_reference']}"
    )


def _read_contingency_from_maturity_by_pp() -> tuple[int, int, int, int]:
    if not MATURITY_BY_PP.exists():
        raise FileNotFoundError(
            f"{MATURITY_BY_PP} not found — run descriptive_inferential.py first "
            "to generate results/statistics/maturity_by_pp.csv."
        )
    df = pd.read_csv(MATURITY_BY_PP)
    pub_row = df[df["Is_Public"].astype(str) == "Public"].iloc[0]
    priv_row = df[df["Is_Public"].astype(str) == "Private"].iloc[0]
    other_cols = [c for c in df.columns if c not in ("Is_Public", "Advanced")]
    pub_adv = int(pub_row["Advanced"]) if "Advanced" in df.columns else 0
    priv_adv = int(priv_row["Advanced"]) if "Advanced" in df.columns else 0
    pub_other = int(sum(pub_row[c] for c in other_cols))
    priv_other = int(sum(priv_row[c] for c in other_cols))
    return pub_adv, pub_other, priv_adv, priv_other


def main() -> int:
    pub_adv, pub_other, priv_adv, priv_other = \
        _read_contingency_from_maturity_by_pp()
    print(f"[IN]  {MATURITY_BY_PP}")
    print(f"      Public Adv={pub_adv} Other={pub_other}; "
          f"Private Adv={priv_adv} Other={priv_other}")
    result = compute_fisher_with_ci(
        public_advanced=pub_adv,
        public_other=pub_other,
        private_advanced=priv_adv,
        private_other=priv_other,
    )
    print_summary(result)
    update_fisher_json(result)
    print(f"\nWrote: {FISHER_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
