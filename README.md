# Curriculum Maturity for AI and Data Science Across Korean Health Professional Schools

Repository: https://github.com/ggg5438/med_school_AI_edu_status

Analysis code for a national cross-sectional document analysis of AI- and
data-science-related coursework across all 63 accredited Korean medical
(n=40), dental (n=11), and Korean medicine (n=12) schools. The study
characterizes how mature each school's AI/DS curriculum is along five content
domains, and analyzes how curriculum maturity varies by profession, governance
(public vs private), and region (Capital area vs Non-Capital area).

The headline finding is structural. Quantitative foundations dominate the AI/DS
coursework, accounting for 62.2% of all credits offered, and are typically
mandatory. AI-specific content is smaller, less consistently offered, and more
often elective. Only 5 of 63 schools (7.9%) reach the Advanced curriculum
stage. Among the 30 schools at the Intermediate stage, 13 (43%) lack only a
mandatory AI-core course — they already meet the credit and breadth thresholds
and could move to Advanced by converting one existing elective into a required
course. Curriculum maturity shows no association with public/private
governance or with Capital-area location.

## What this code does

Starting from a frozen, human-adjudicated course classification (each course
labelled across five domains) and the raw institutional metadata, the pipeline
reproduces every statistic, table, and figure reported in the manuscript:

- Domain credit distribution and school-level adoption rates (denominator n=63)
- Course-level mandatory share by domain
- Curriculum maturity classification across four stages (None,
  Foundational-Only, Intermediate, Advanced) using the Option B Advanced
  definition
- Per-school gap analysis: credits, AI-core domains, and mandatory AI-core
  courses needed to reach the next stage
- Stratified mandatory gap (quantitative foundations vs AI-core) by
  profession, governance, and region
- Friedman test with pairwise Wilcoxon (Holm-Bonferroni) on domain credits
- Kruskal-Wallis on total credits across professions
- Fisher exact odds ratios with exact conditional confidence intervals for
  Advanced stage by governance and by region
- BCa bootstrap confidence intervals for domain adoption (College-stratified)
- Threshold sweep for the Advanced credit cutoff (5 to 12 credits)
- Five-strategy classification sensitivity analysis (consensus, single-domain
  priority, strict single-domain, rule-based only, LLM-assisted only)
- Advanced-vs-Foundational-Only extreme-group profiling
- Ordinal logistic regression of the 4-stage maturity outcome on profession,
  governance, region, and admission quota
- Three main figures plus a per-profession supplementary figure

## Five content domains

- **D1** Quantitative Foundations (mathematics, statistics, programming,
  computational thinking)
- **D2** AI and Machine Learning
- **D3** Data Science
- **D4** Health Informatics
- **D5** Clinical AI Application

D2, D3, and D5 together constitute the AI-core domains.

## Maturity stages (Option B Advanced)

A school's maturity is assigned in four ordered levels:

- **None**: no AI/DS course offered
- **Foundational-Only**: at least one course, but no AI-core domain
- **Intermediate**: at least one AI-core domain
- **Advanced**: D1 offered, at least two AI-core domains, at least eight total
  AI/DS credits, and at least one mandatory AI-core course

The Advanced definition (Option B) adds the mandatory AI-core requirement on
top of the breadth (D1 + ≥2 AI-core domains) and depth (≥8 credits)
thresholds. This captures schools that have institutionalized AI/DS content
rather than offered it only as a peripheral elective.

## Repository layout

```text
release_github/
├── README.md
├── requirements.txt
├── LICENSE                  GNU General Public License v3.0 (GPL-3.0)
├── run_all.py               End-to-end orchestrator
└── src/
    ├── analysis.py                Primary analysis (per-school features,
    │                              4-stage maturity, sensitivity, threshold
    │                              sweep, bootstrap, gap analysis,
    │                              stratified mandatory gap)
    ├── supplementary_analysis.py  Advanced vs Foundational profiling and
    │                              ordinal logistic regression
    └── figures.py                 Figures 1, 2, 3 + per-profession
                                   supplementary figure
```

## Input data

**Input data are NOT included in this repository.** They are available from
the corresponding author on reasonable request. The dataset was assembled
from publicly available institutional materials (course catalogues, academic
schedules, and curriculum guides) collected between April and June 2025, and
contains no personal or sensitive information.

To run the pipeline, place the input files under `data/` using this layout:

```text
data/
├── raw/
│   ├── 교육과정현황조사 최종본.xlsx   Course-level curriculum survey
│   └── 대학정보.xlsx                   Institutional metadata
└── classification/
    ├── final_adjudicated_classification.csv  Primary frozen classification
    ├── m1_rules.csv                          Rule-based keyword classification
    └── m2_llm.csv                            LLM-assisted classification
```

The classification files contain per-course domain labels (`D1`–`D5`). They
are treated as a frozen input: the classification is not regenerated by this
code. The `m1_rules.csv` and `m2_llm.csv` files are used only by the
five-strategy classification sensitivity analysis.

## Setup

Tested on Python 3.13.

```bash
pip install -r requirements.txt
```

## Run

After placing the input data under `data/`:

```bash
python run_all.py
```

The orchestrator runs the three stages in dependency order (primary analysis
writes the per-school feature table that the supplementary analysis and the
figures both read).

Individual stages can also be run directly:

```bash
python src/analysis.py
python src/supplementary_analysis.py
python src/figures.py
```

## Outputs

Running the pipeline creates and populates two directories:

```text
results/statistics/
├── per_school.csv                     63-school feature table
├── summary.json                        Headline numbers
├── gap_analysis.csv                    Per-school gap to next stage
├── mandatory_gap_stratified.csv        D1 vs AI-core mandatory by stratum
├── bootstrap_adoption_bca.csv          BCa CIs for domain adoption
├── threshold_sweep.csv                 Advanced credit threshold sweep
├── sensitivity.csv                     5-strategy classification sensitivity
├── adv_vs_foundational.csv             Extreme-group profiling
├── ordinal_logistic.csv                Ordinal logistic coefficients
└── ordinal_logistic_summary.json       Fit statistics

figures/
├── Figure_1.{png,pdf}
├── Figure_2.{png,pdf}
├── Figure_3.{png,pdf}
└── Supplementary_Note_5_PerProfession.{png,pdf}
```

All numerical analyses use a fixed random seed (42), so re-running reproduces
the reported numbers and figures exactly.

## Citation

> [Authors. Title. Journal. Year.]

## License

Code is released under the GNU General Public License v3.0 (GPL-3.0); see
`LICENSE`. The curriculum dataset and derived data products are not part of
this repository and are governed separately.
