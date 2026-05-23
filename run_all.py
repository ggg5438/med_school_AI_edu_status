# End-to-end reproduction orchestrator.
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"

STAGES = [
    ("descriptive_inferential.py", "Descriptive + inferential statistics"),
    ("domain_credit_tests.py", "Friedman + pairwise Wilcoxon on domain credits"),
    ("region_analysis.py", "Capital vs Non-Capital area comparison"),
    ("school_type_axes.py", "Four-axis school-type decomposition"),
    ("threshold_sweep.py", "Advanced-threshold sensitivity sweep"),
    ("gap_stratified.py", "Stratified mandatory gap"),
    ("power_analysis.py", "Post-hoc power / MDE + threshold justification"),
    ("fisher_ci.py", "Governance Fisher OR with exact CI"),
    ("bootstrap_ci.py", "BCa bootstrap CIs for headline effect sizes"),
    ("ordinal_logistic.py", "Exploratory ordinal logistic regression"),
    ("advanced_vs_foundational.py", "Advanced vs Foundational-Only profiling"),
    ("sensitivity_full.py", "Full classification sensitivity"),
    ("figures.py", "Figures 1/2/3 + per-profession supplementary"),
]


def run(script: str) -> None:
    print(f"\n{'=' * 70}\n[run] {script}\n{'=' * 70}", flush=True)
    result = subprocess.run([sys.executable, str(SRC_DIR / script)])
    if result.returncode != 0:
        raise SystemExit(
            f"[run_all] {script} failed with exit code {result.returncode}."
        )


def main() -> int:
    t0 = time.time()
    print("=" * 74)
    print("REPRODUCTION PIPELINE")
    print("=" * 74)
    for script, label in STAGES:
        print(f"\n>>> {label}")
        run(script)
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
