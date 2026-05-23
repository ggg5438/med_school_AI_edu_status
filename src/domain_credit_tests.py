# Domain credit difference: Friedman + pairwise Wilcoxon post-hoc.
from __future__ import annotations

import json
import sys
from itertools import combinations
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

DOMAIN_CREDIT_COLS = [
    "D1_Quantitative_Foundations_Credits",
    "D2_AI_ML_Credits",
    "D3_Data_Science_Credits",
    "D4_Health_Informatics_Credits",
    "D5_Clinical_AI_Application_Credits",
]


def kendall_w_from_friedman(chi2: float, n: int, k: int) -> float:
    return float(chi2 / (n * (k - 1)))


def main() -> int:
    print("=" * 70)
    print("DOMAIN CREDIT DIFFERENCE — Friedman + pairwise Wilcoxon")
    print("=" * 70)
    course_df, uni_info, uni_df = load_all(classifier="consensus")

    mat = uni_df[DOMAIN_CREDIT_COLS].values
    n, k = mat.shape

    chi2, p = sstats.friedmanchisquare(*[mat[:, j] for j in range(k)])
    W = kendall_w_from_friedman(chi2, n, k)
    print(f"\nFriedman: chi^2 = {chi2:.3f}, df = {k-1}, p = {p:.3e}")
    print(f"Kendall's W = {W:.3f}")

    pair_rows = []
    pair_pvals = []
    for i, j in combinations(range(k), 2):
        a = mat[:, i]
        b = mat[:, j]
        d = a - b
        try:
            stat, p_pair = sstats.wilcoxon(
                a, b, zero_method="wilcox", correction=False
            )
        except ValueError:
            stat, p_pair = float("nan"), 1.0
        d_nonzero = d[d != 0]
        if len(d_nonzero) == 0:
            r = float("nan")
        else:
            ranks = sstats.rankdata(np.abs(d_nonzero))
            w_plus = float(np.sum(ranks[d_nonzero > 0]))
            w_minus = float(np.sum(ranks[d_nonzero < 0]))
            denom = w_plus + w_minus
            r = (w_plus - w_minus) / denom if denom > 0 else float("nan")
        pair_rows.append({
            "Domain_A": DOMAIN_CREDIT_COLS[i].replace("_Credits", ""),
            "Domain_B": DOMAIN_CREDIT_COLS[j].replace("_Credits", ""),
            "Mean_A": float(np.mean(a)),
            "Mean_B": float(np.mean(b)),
            "Median_A": float(np.median(a)),
            "Median_B": float(np.median(b)),
            "Wilcoxon_W": float(stat),
            "p_value": float(p_pair),
            "rank_biserial_r": r,
            "abs_rank_biserial_r": abs(r) if not np.isnan(r) else float("nan"),
        })
        pair_pvals.append(p_pair)

    pair_pvals = np.array(pair_pvals)
    order = np.argsort(pair_pvals)
    m = len(pair_pvals)
    holm = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pair_pvals[idx] * (m - rank)
        running_max = max(running_max, min(adj, 1.0))
        holm[idx] = running_max
    for i, h in enumerate(holm):
        pair_rows[i]["p_holm"] = float(h)
        pair_rows[i]["significant_holm"] = bool(h < 0.05)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT_DIR / "domain_pairwise_wilcoxon.csv",
                   index=False, encoding="utf-8-sig")
    print("\n[Pairwise Wilcoxon (Holm-adjusted)]")
    print(pair_df[[
        "Domain_A", "Domain_B", "Median_A", "Median_B",
        "rank_biserial_r", "p_value", "p_holm", "significant_holm"
    ]].to_string(index=False))

    summary = {
        "n_schools": int(n),
        "n_domains": int(k),
        "domains": [c.replace("_Credits", "") for c in DOMAIN_CREDIT_COLS],
        "friedman": {
            "chi_squared": float(chi2),
            "df": int(k - 1),
            "p_value": float(p),
            "kendall_W": float(W),
        },
        "pairwise_method": "Wilcoxon signed-rank, two-sided, "
                           "Holm-Bonferroni within 10-pair family",
    }
    with open(OUT_DIR / "domain_friedman.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {OUT_DIR / 'domain_friedman.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
