from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

STAGES = [
    ("analysis.py", "Primary analysis (Option B Advanced, 4-stage maturity)"),
    ("supplementary_analysis.py", "Supplementary analysis (Adv vs Foundational, ordinal logistic)"),
    ("figures.py", "Figures (1, 2, 3) + Supplementary Note 5"),
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
    print("=" * 70)
    print("REPRODUCTION PIPELINE")
    print("=" * 70)

    (ROOT / "results" / "statistics").mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)

    for script, label in STAGES:
        print(f"\n>>> {label}")
        run(script)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
