# Curriculum Maturity for AI and Data Science Across Korean Health Professional Schools

Repository: https://github.com/ggg5438/med_school_AI_edu_status

Analysis code for a national cross-sectional document analysis of AI- and
data-science-related coursework across Korean medical, dental, and Korean
medicine schools. The study characterizes how mature each school's AI/data
science curriculum is along five content domains, and analyzes how curriculum
maturity varies by profession, governance (public vs private), and region. The
cohort comprises 179 AI/DS-related courses across 60 schools (38 medical, 11
dental, 11 Korean medicine). The headline finding is structural: quantitative
foundations are widely adopted and largely mandatory, while applied AI content
is smaller, less consistently offered, and more often elective.

## What this code does

Starting from a frozen, human-adjudicated course classification (each course
labelled across five domains) and the raw curriculum and institutional
metadata, the pipeline reproduces every statistic, table, and figure reported in
the manuscript:

- Domain credit distribution and school-level adoption rates
- Mandatory-vs-elective ratios by domain
- Curriculum maturity stages (Foundational-Only / Intermediate / Advanced) and
  the gap to the next stage
- Descriptive Table 1
- Friedman test with pairwise Wilcoxon (Holm-Bonferroni) on domain credits
- Kruskal-Wallis (profession), Mann-Whitney (governance, region)
- Fisher exact odds ratios with exact conditional confidence intervals
  (governance and region vs Advanced stage)
- BCa bootstrap confidence intervals (domain adoption; regional mean differences)
- Post-hoc power / minimum-detectable-effect analysis and an empirical
  justification for the Advanced credit threshold
- Advanced-threshold sensitivity sweep and a five-strategy classification
  sensitivity analysis
- Exploratory ordinal logistic regression of curriculum maturity
- Advanced-vs-Foundational-Only extreme-group profiling
- Stratified mandatory gap (by college, governance, and region)
- Figures 1, 2, and 3, plus a per-profession supplementary figure

## Five domains

- **D1** Quantitative Foundations
- **D2** AI and Machine Learning
- **D3** Data Science
- **D4** Health Informatics
- **D5** Clinical AI Application

AI-core domains are D2, D3, and D5. A school is classified **Advanced** if it
offers a foundational (D1) course, at least two AI-core domains, and at least 8
total AI/DS credits.

## Repository layout

```text
release_github/
├── README.md
├── requirements.txt
├── LICENSE                       GNU General Public License v3.0 (GPL-3.0)
├── run_all.py                    End-to-end orchestrator
└── src/
    ├── config.py                 Paths, mappings, domain definitions, palettes
    ├── data_loader.py            Raw-data preprocessing, feature matrix, maturity
    ├── classification_loader.py  Merge frozen classification with raw curriculum
    ├── analysis.py               Core descriptive + inferential statistics
    ├── descriptive_inferential.py  Stage 1 entry point (+ classification sensitivity)
    ├── domain_credit_tests.py    Friedman + pairwise Wilcoxon on domain credits
    ├── region_analysis.py        Capital vs Non-Capital area comparison
    ├── school_type_axes.py       Four-axis school-type decomposition
    ├── threshold_sweep.py        Advanced-threshold sensitivity sweep
    ├── gap_stratified.py         Stratified mandatory gap
    ├── power_analysis.py         Post-hoc power / MDE + threshold justification
    ├── fisher_ci.py              Governance Fisher OR with exact conditional CI
    ├── bootstrap_ci.py           BCa bootstrap CIs for headline effect sizes
    ├── ordinal_logistic.py       Exploratory ordinal logistic regression
    ├── advanced_vs_foundational.py  Advanced vs Foundational-Only profiling
    ├── sensitivity_full.py       Full five-strategy classification sensitivity
    └── figures.py                Figures 1/2/3 + per-profession supplementary
```

## Input data

**Input data are NOT included in this repository.** They are available from the
corresponding author on reasonable request. The dataset was assembled from
publicly available institutional materials (course catalogues, academic
schedules, and curriculum guides) collected between April and June 2025, and
contains no personal or sensitive information.

To run the pipeline, place the input files under `data/` using this layout:

```text
data/
├── raw/
│   ├── 교육과정현황조사 최종본.xlsx    Course-level curriculum survey
│   └── 대학정보.xlsx                    Institutional metadata (region, quota, governance)
└── classification/
    ├── consensus.csv                    Primary adjudicated consensus classification
    ├── final_adjudicated_classification.csv  Final adjudicated classification
    ├── m1_rules.csv                     Rule-based keyword classification (sensitivity)
    └── m2_llm.csv                       LLM-assistant classification (sensitivity)
```

The classification files contain per-course domain labels (`D1`–`D5`). They are
treated as a frozen input: the classification is not regenerated by this code.
The adjudicated consensus and the final adjudicated classification carry the
same per-course domain assignments; both are listed because different pipeline
stages read one or the other.

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

The orchestrator runs all stages in dependency order. Some stages read CSVs
written by earlier stages (e.g. the figures read derived statistics, and the
Fisher CI stage reads the governance contingency table), so run them through
`run_all.py` rather than individually unless the prerequisites already exist.

Individual stages can also be run directly, for example:

```bash
python src/descriptive_inferential.py
python src/figures.py
```

## Outputs

Running the pipeline creates (and is the sole source of) two directories:

```text
results/
├── statistics/    All statistical CSV/JSON outputs reported in the manuscript
└── sensitivity/   Classification-strategy sensitivity summary + bootstrap CIs
figures/
├── Figure_1.{png,pdf}
├── Figure_2.{png,pdf}
├── Figure_3.{png,pdf}
└── Supplementary_PerProfession.{png,pdf}
```

All numerical analyses use a fixed random seed (42), so re-running reproduces the
reported numbers and figures exactly.

## Citation

> [Authors. Title. Journal. Year.]

## License

Code is released under the GNU General Public License v3.0 (GPL-3.0); see
`LICENSE`. The curriculum dataset and derived data products are not part of this
repository and are governed separately.
