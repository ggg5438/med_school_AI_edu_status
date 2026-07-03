# AI and Data Science Coursework Across Korean Health Professional Schools

Repository: https://github.com/ggg5438/med_school_AI_edu_status

Analysis code for a national cross-sectional document analysis of AI- and
data-science-related coursework across all 63 accredited Korean medical
(n=40), dental (n=11), and Korean medicine (n=12) schools. The study measures
what AI/DS content each school actually offers along five content domains and
analyzes how that content varies by profession, governance (public vs private),
and region (Capital area vs non-Capital area).

The picture is measurement-first and structural. Quantitative foundations
dominate the AI/DS coursework, accounting for 62.2% of all credits offered and
usually mandatory. AI-specific content (AI/ML, data science, clinical AI) is
smaller, offered by fewer schools, and more often elective. When AI/DS content
is decomposed into a common quantitative baseline and two career tracks
(clinical vs research), schools accumulate the two tracks asymmetrically and
mostly in the pre-clinical years.

## What this code does

Starting from a frozen, human-adjudicated course classification (each course
labeled across five domains) and the raw institutional metadata, the pipeline
reproduces the statistics and figures reported in the manuscript:

- Domain credit distribution and school-level adoption rates (denominator n=63)
- Course-level mandatory share by domain (chi-square = 29.5)
- Friedman test with pairwise Wilcoxon (Holm-Bonferroni) on the five domain
  credits (chi-square = 92.66, Kendall's W = 0.368)
- Kruskal-Wallis on total credits across professions
- A reference-configuration measurement (a school offers quantitative
  foundations, covers at least two AI-specific domains, and reaches at least
  eight AI/DS credits), with Fisher exact tests by governance and region
- Career-track metrics for the common baseline, clinical, and research tracks,
  including year-weighted credit exposure (clinical-proximity, linear, and
  clinical-only weight functions), late-stage exposure, and track co-occurrence
- Multivariable OLS of per-school total credits on institutional axes, backing
  Figure 4a (public vs private adjusted beta = 2.62, 95% CI 0.18 to 5.06)
- Multivariable track-holding logistic regression on institutional axes,
  backing Figure 4b (Capital-area effect on holding the research track,
  odds ratio = 2.99, 95% CI 0.84 to 10.60)
- Capital vs non-Capital Mann-Whitney U with Holm-Bonferroni correction
  (data science credits, Holm-adjusted P = .010)
- BCa bootstrap confidence intervals for domain adoption (College-stratified)
- Classification-strategy sensitivity (five course-classification strategies:
  consensus, single-domain priority, strict single-domain, rule-based only,
  LLM-assisted only)
- Retrospective power / minimum detectable effect for the fixed national cohort
- Extreme-group profiling by total-credit quartiles and by AI-specific breadth
- Four main figures plus a per-profession supplementary figure

## Five content domains

- **D1** Quantitative Foundations (mathematics, statistics, programming,
  computational thinking)
- **D2** AI and Machine Learning
- **D3** Data Science
- **D4** Health Informatics
- **D5** Clinical AI Application

D2, D3, and D5 together constitute the AI-specific domains. The two career
tracks group the applied domains: the clinical track is D4 + D5 and the
research track is D2 + D3, with D1 as a common quantitative baseline.

## Repository layout

```text
release_github/
├── README.md
├── requirements.txt
├── LICENSE                  GNU General Public License v3.0 (GPL-3.0)
├── run_all.py               End-to-end orchestrator
└── src/
    ├── analysis.py                Primary analysis (per-school features,
    │                              domain distribution, mandatory share,
    │                              Friedman/Wilcoxon, track metrics, BCa
    │                              bootstrap, classification sensitivity)
    ├── supplementary_analysis.py  Extreme-group profiling, continuous-outcome
    │                              OLS, track-holding logistic and depth OLS,
    │                              year-weight sensitivity
    ├── region.py                  Capital vs non-Capital Mann-Whitney (Holm)
    ├── power.py                   Retrospective power / minimum detectable effect
    └── figures.py                 Figures 1-4 + per-profession supplementary figure
```

## Input data

**Input data are NOT included in this repository.** The curriculum data were
drawn from publicly accessible Korean medical, dental, and Korean-medicine
school websites (course catalogues, academic schedules, and curriculum guides)
collected between April and June 2025. Because those institutional materials
are under institutional copyright, they were obtained directly from the school
websites and are not redistributed here. The structured classification dataset
that this code consumes is available from the corresponding author on
reasonable request. The data contain no personal or sensitive information.

To run the pipeline, place the input files under `data/` using this layout:

```text
data/
├── raw/
│   ├── 교육과정현황조사 최종본.xlsx   Course-level curriculum survey
│   └── 대학정보.xlsx                   Institutional metadata (63 schools)
└── classification/
    ├── final_adjudicated_classification.csv  Primary frozen classification
    ├── m1_rules.csv                          Rule-based keyword classification
    └── m2_llm.csv                            LLM-assisted classification
```

The classification files carry per-course domain labels (`D1`–`D5`). They are
treated as frozen inputs; the classification is not regenerated by this code.
The `m1_rules.csv` and `m2_llm.csv` files are used only by the
classification-strategy sensitivity analysis.

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

The orchestrator runs the stages in dependency order: the primary analysis
writes the per-school feature table and track metrics that the supplementary
analysis, region test, power analysis, and figures all read; the figures
additionally read the supplementary regression tables.

Individual stages can also be run directly:

```bash
python src/analysis.py
python src/supplementary_analysis.py
python src/region.py
python src/power.py
python src/figures.py
```

## Outputs

Running the pipeline creates and populates two directories:

```text
results/statistics/
├── per_school.csv                      63-school feature table
├── summary.json                        Headline numbers
├── track_metrics.csv                   Per-school career-track metrics
├── track_summary.json                  Track distribution + co-occurrence
├── mandatory_gap_stratified.csv        D1 vs AI-specific mandatory by stratum
├── bootstrap_adoption_bca.csv          BCa CIs for domain adoption
├── sensitivity.csv                     5-strategy classification sensitivity
├── quartile_extremes_by_total_credits.csv   Extreme-group profiling (credits)
├── quartile_extremes_by_aicore_count.csv    Extreme-group profiling (breadth)
├── continuous_ols_total_credits.csv    OLS backing Figure 4a (+ _summary.json)
├── continuous_ols_aicore_count.csv     OLS on AI-specific breadth (+ _summary.json)
├── track_presence_logistic.csv         Track-holding logistic backing Figure 4b
├── track_depth_ols.csv                 Conditional track-depth OLS
├── track_weight_sensitivity.csv        Year-weight function comparison
├── track_wilcoxon_by_weight.csv        Clinical vs research by weight function
├── track_rank_correlation.csv          Weight-function rank agreement
├── region_mwu.csv                      Capital vs non-Capital Mann-Whitney (Holm)
└── power_analysis.json                 Retrospective power / MDE

figures/
├── Figure_1.{png,pdf}
├── Figure_2.{png,pdf}
├── Figure_3.{png,pdf}
├── Figure_4.{png,pdf}
└── Supplementary_Note_1_PerProfession.{png,pdf}
```

All resampling-based analyses use a fixed random seed (42), so re-running the
pipeline reproduces the reported numbers and figures exactly.

## Citation

> Jang D, Shin J, Kim C. Documented Curriculum Exposure to AI and Data Science Across Korean Medical, Dental, and Korean Medicine Schools: A National Cross-Sectional Document Analysis. Manuscript submitted for publication; 2026.

## License

Code is released under the GNU General Public License v3.0 (GPL-3.0); see
`LICENSE`. The curriculum dataset and derived data products are not part of
this repository and are governed separately.
