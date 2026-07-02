# -*- coding: utf-8 -*-
"""
Main figures (1, 2, 3, 4) + Supplementary Note 1.

  Figure 1  Quantitative-foundation-centric curriculum: domain total credits,
            credits per school by profession, domain adoption heatmap, and
            year x domain course placement.
  Figure 2  AI-specific institutionalization (2 panels): mandatory share by
            domain and an institutionalization map (adoption x mandatory share,
            sized by credit share).
  Figure 3  Clinical vs research career-track distribution (3 panels): per-school
            year-weighted track credits, mean crude credits by profession x
            track (95% bootstrap CI), and track co-occurrence counts.
  Figure 4  Institutional-characteristic axes (2-panel forest): multivariable
            OLS beta (95% CI) for per-school total credits, and multivariable
            logistic OR (95% CI) for holding each career track. Both significant
            and non-significant terms are drawn identically. Visualization only:
            every value is read straight from the statistics CSVs.
  Supp Note 1  Per-profession year x domain course placement (6 heatmaps).

Inputs (results/statistics/ + data/):
  results/statistics/per_school.csv
  results/statistics/summary.json
  results/statistics/track_metrics.csv
  results/statistics/track_summary.json
  results/statistics/track_presence_logistic.csv
  results/statistics/continuous_ols_total_credits.csv
  data/classification/final_adjudicated_classification.csv
  data/raw/교육과정현황조사 최종본.xlsx

Output:
  figures/Figure_1.{png,pdf}
  figures/Figure_2.{png,pdf}
  figures/Figure_3.{png,pdf}
  figures/Figure_4.{png,pdf}
  figures/Supplementary_Note_1_PerProfession.{png,pdf}

Style: publication style, Arial/sans-serif, colorblind-safe.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

PROJECT_ROOT = _SCRIPTS_DIR.parent
DATA_CLASSIF = PROJECT_ROOT / "data" / "classification" / "final_adjudicated_classification.csv"
STATS_DIR = PROJECT_ROOT / "results" / "statistics"
RAW_CURR_FILE = PROJECT_ROOT / "data" / "raw" / "교육과정현황조사 최종본.xlsx"

OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Style
# ============================================================

def setup_style():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "axes.unicode_minus": False,
    })


# Page sizes (mm -> inches)
SINGLE_COL = 89 / 25.4
ONE_HALF_COL = 136 / 25.4
DOUBLE_COL = 183 / 25.4


def _panel_label(ax, label, x=-0.10, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def _save(fig, stem: str) -> None:
    for fmt in ("png", "pdf"):
        out = OUT_DIR / f"{stem}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05,
                    facecolor="white", edgecolor="none")
        print(f"    Saved: {out.relative_to(PROJECT_ROOT)}")
    plt.close(fig)


# ============================================================
# Constants
# ============================================================

COLLEGE_MAP = {"의대": "Medicine", "치대": "Dentistry", "한의대": "Korean Medicine"}
COLLEGE_ORDER = ["Medicine", "Dentistry", "Korean Medicine"]
COLLEGE_SHORT = {"Medicine": "Medical", "Dentistry": "Dental", "Korean Medicine": "Korean medicine"}

DOMAIN_ORDER = [
    "Quantitative Foundations",
    "AI & Machine Learning",
    "Health Informatics",
    "Data Science",
    "Clinical AI Application",
]
DOMAIN_SHORT = {
    "Quantitative Foundations": "Quantitative\nfoundations",
    "AI & Machine Learning": "AI &\nmachine learning",
    "Data Science": "Data\nscience",
    "Health Informatics": "Health\ninformatics",
    "Clinical AI Application": "Clinical AI\napplication",
}
DOMAIN_ABBR = {
    "Quantitative Foundations": "Quant",
    "AI & Machine Learning": "AI/ML",
    "Data Science": "DS",
    "Health Informatics": "HI",
    "Clinical AI Application": "ClinAI",
}
DOMAIN_CSVCOL = {
    "Quantitative Foundations": "D1",
    "AI & Machine Learning": "D2",
    "Data Science": "D3",
    "Health Informatics": "D4",
    "Clinical AI Application": "D5",
}

DOMAIN_COLORS = {
    "Quantitative Foundations": "#D65F5F",
    "AI & Machine Learning":    "#4878D0",
    "Data Science":             "#956CB4",
    "Health Informatics":       "#8C613C",
    "Clinical AI Application":  "#DC7EC0",
}
FOUNDATIONAL_COLOR = "#D65F5F"
AI_SPECIFIC_COLOR = "#4878D0"

COLLEGE_COLORS = {
    "Medicine":        "#4878D0",
    "Dentistry":       "#EE854A",
    "Korean Medicine": "#6ACC64",
}

# Track colour mapping (stable across Figure 3 and the supplementary figure)
TRACK_COLORS = {
    "clinical": "#1f77b4",
    "research": "#d62728",
    "baseline": "#7f7f7f",
}
TRACK_LABELS = {
    "clinical": "Clinical (D4+D5)",
    "research": "Research (D2+D3)",
    "baseline": "Baseline (D1)",
}

GOV_COLORS = {"Public": "#4878D0", "Private": "#D4D4D4"}
GRID_COLOR = "#D9D9D9"

# ------------------------------------------------------------------
# Uniform thin grey edge applied to EVERY filled mark (bars, stacked-bar
# segments, heatmap cells, scatter / bubble markers) across all panels and
# all figures, per the visual-consistency rule. A single constant keeps
# the stroke width identical for every graphic type (heatmaps included), so
# no panel looks edged while another looks borderless.
# ------------------------------------------------------------------
EDGE_LW = 0.5          # points; identical for bars, cells and markers
EDGE_COLOR = "#999999"  # mid grey, legible on white without overpowering fills


def _draw_cell_grid(ax, nrows: int, ncols: int) -> None:
    """Overlay a uniform thin grey border on every cell of an imshow heatmap.

    imshow places cell centres at integer (col, row); each cell spans
    [j-0.5, j+0.5] x [i-0.5, i+0.5]. We stroke a facecolor='none' rectangle
    per cell so the grid is identical for filled, masked and zero cells.
    """
    for i in range(nrows):
        for j in range(ncols):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="none",
                                       edgecolor=EDGE_COLOR,
                                       linewidth=EDGE_LW, zorder=5))


def _bootstrap_ci_mean(vals, rng, n_boot: int = 10000, ci: float = 95.0):
    """Percentile bootstrap CI for the mean of a non-negative sample.

    Because every resample is drawn from non-negative observations, every
    bootstrap mean is >= 0, so the lower bound is bounded below by 0 by
    construction (no negative lower limit can arise).
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    if n < 2:
        return mean, mean, mean
    boot = rng.choice(vals, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, (100.0 - ci) / 2.0))
    hi = float(np.percentile(boot, 100.0 - (100.0 - ci) / 2.0))
    return mean, lo, hi


# ============================================================
# Data loading
# ============================================================

def load_classification() -> pd.DataFrame:
    """Course-level data (179 courses) from the v7 adjudicated classification."""
    df = pd.read_csv(DATA_CLASSIF)
    df["College_EN"] = df["College"].map(COLLEGE_MAP)
    df["Is_Mandatory_Binary"] = (df["Is_Mandatory"] == "필수").astype(int)
    return df


def load_school_v11() -> pd.DataFrame:
    """School-level data (n=63) loaded directly from per_school.csv."""
    # keep_default_na=False keeps string columns literal; numeric columns are
    # coerced explicitly below.
    df = pd.read_csv(STATS_DIR / "per_school.csv", keep_default_na=False)
    # Restore numeric columns
    for c in ["n_courses", "total_credits",
              "D1_n", "D1_credits", "has_D1",
              "D2_n", "D2_credits", "has_D2",
              "D3_n", "D3_credits", "has_D3",
              "D4_n", "D4_credits", "has_D4",
              "D5_n", "D5_credits", "has_D5",
              "Admission_Quota", "Capital_Area",
              "aicore_count", "has_mandatory_aicore"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Governance: 국립/공립 -> Public, 사립 -> Private
    pp_map = {"국립": "Public", "공립": "Public", "사립": "Private"}
    df["Governance"] = df["Public_Private"].map(pp_map)
    # Region label (Capital vs non-Capital)
    df["Region_Binary"] = df["Capital_Area"].map({1: "Capital area",
                                                  0: "Non-Capital area"})
    # AI-core count and breadth (for Fig 2c)
    df["AI_Core_Count"] = df[["has_D2", "has_D3", "has_D5"]].sum(axis=1)
    df["Breadth"] = df[["has_D1", "has_D2", "has_D3", "has_D4", "has_D5"]].sum(axis=1)
    # Total credits column for compatibility with Fig 1b
    df = df.rename(columns={"total_credits": "Total_Credits",
                            "College": "College_EN"})
    return df


def load_summary_v11() -> dict:
    with open(STATS_DIR / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_track_metrics_v11() -> pd.DataFrame:
    return pd.read_csv(STATS_DIR / "track_metrics.csv")


def load_track_summary_v11() -> dict:
    with open(STATS_DIR / "track_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_track_presence_logistic_v11() -> pd.DataFrame:
    return pd.read_csv(STATS_DIR / "track_presence_logistic.csv")


def load_continuous_ols_total_credits_v11() -> pd.DataFrame:
    return pd.read_csv(STATS_DIR / "continuous_ols_total_credits.csv")


# ============================================================
# Course-level derived stats (recomputed from 179 courses)
# ============================================================

def compute_domain_distribution(course_df: pd.DataFrame,
                                school_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_credits_all = float(course_df["Credits"].sum())
    n_schools_total = len(school_df)
    for dom in DOMAIN_ORDER:
        col = DOMAIN_CSVCOL[dom]
        dom_courses = course_df[course_df[col] == 1]
        total = float((dom_courses["Credits"]).sum())
        n_offering = int((school_df[f"has_{col}"] >= 1).sum())
        rows.append({
            "Domain": dom,
            "Total_Credits": total,
            "Pct_Credits": 100.0 * total / total_credits_all,
            "N_Schools": n_offering,
            "Pct_Schools": 100.0 * n_offering / n_schools_total,
        })
    return pd.DataFrame(rows).set_index("Domain").loc[DOMAIN_ORDER].reset_index()


def compute_mandatory_by_domain(course_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dom in DOMAIN_ORDER:
        col = DOMAIN_CSVCOL[dom]
        sub = course_df[course_df[col] == 1]
        n = len(sub)
        nm = int(sub["Is_Mandatory_Binary"].sum())
        rows.append({
            "Domain": dom,
            "N_Courses": n,
            "N_Mandatory": nm,
            "Pct_Mandatory": 100.0 * nm / n if n > 0 else 0.0,
        })
    return pd.DataFrame(rows).set_index("Domain").loc[DOMAIN_ORDER].reset_index()


# ============================================================
# Year stage helpers (unchanged from v10)
# ============================================================

def _load_year_merged(course_df: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_excel(RAW_CURR_FILE)
    raw.columns = [
        "College", "University", "Course_Name", "Prog_SW", "Math_Stat",
        "Informatics", "AI", "DataSci", "Pre_Medical", "Medical",
        "Is_Mandatory", "Year_Raw", "Type", "Liberal", "Major_Required",
        "Major_Elective", "Credits", "Theory_Hours", "Practice_Hours",
        "Source", "Note",
    ]
    raw = raw.reset_index().rename(columns={"index": "_row_idx"})
    raw["Course_ID"] = raw["_row_idx"] + 1

    stage_map = {
        "예 1-1": "Pre-1", "예 1-2": "Pre-1", "예 1": "Pre-1",
        "예1": "Pre-1", "예1-1": "Pre-1", "예1-2": "Pre-1",
        "예 2-1": "Pre-2", "예 2-2": "Pre-2", "예 2": "Pre-2",
        "예2": "Pre-2", "예2-1": "Pre-2", "예2-2": "Pre-2",
        "예과 1학기": "Pre-1", "예과 2학기": "Pre-2",
        "본 1-1": "Med-1", "본 1-2": "Med-1", "본과 1-2": "Med-1",
        "본1": "Med-1", "본1-1": "Med-1", "본1-2": "Med-1",
        "본 2-1": "Med-2", "본 2-2": "Med-2",
        "본2": "Med-2", "본2-1": "Med-2", "본3": "Med-3",
        "본 3-1": "Med-3", "본 3-2": "Med-3", "본 3": "Med-3",
        "본 4-1": "Med-4", "본 4-2": "Med-4", "본 4": "Med-4", "본4": "Med-4",
    }

    def parse_year(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        for vague in ("예과", "본과", "전학년", "대학원"):
            if s.startswith(vague) and len(s.replace(" ", "")) <= len(vague) + 3:
                return None
        if "~" in s or "," in s:
            return None
        if "본과1-4" in s.replace(" ", ""):
            return None
        if s in stage_map:
            return stage_map[s]
        if "학사과정" in s:
            for d in ("2", "3", "4"):
                if d in s:
                    v = int(d)
                    return f"Med-{v - 1}" if v <= 4 else None
        return None

    raw["Year_Stage"] = raw["Year_Raw"].apply(parse_year)
    merged = course_df.merge(
        raw[["Course_ID", "Year_Stage", "Pre_Medical", "Medical"]],
        on="Course_ID", how="left",
    )
    return merged


def _course_level_stage_share_resolved(course_df: pd.DataFrame,
                                       college_filter: str | None = None):
    """Return (pct_df, counts_df).

    counts_df holds the raw per-(domain, stage) course counts that feed each
    cell percentage; the per-row denominator is counts_df.sum(axis=1) (the
    number of grade-resolved courses in that domain). Returning the source
    counts lets cell annotations show "%(numerator/denominator)" without any
    new counting logic.
    """
    merged = _load_year_merged(course_df)
    merged = merged[merged["Year_Stage"].notna()].copy()
    if college_filter is not None:
        merged = merged[merged["College_EN"] == college_filter].copy()

    stages = ["Pre-1", "Pre-2", "Med-1", "Med-2", "Med-3", "Med-4"]
    counts = np.zeros((len(DOMAIN_ORDER), len(stages)))
    for i, dom in enumerate(DOMAIN_ORDER):
        col = DOMAIN_CSVCOL[dom]
        sub = merged[merged[col] == 1]
        for _, row in sub.iterrows():
            ys = row["Year_Stage"]
            if isinstance(ys, str) and ys in stages:
                counts[i, stages.index(ys)] += 1

    pct = np.full_like(counts, np.nan, dtype=float)
    for i in range(counts.shape[0]):
        row_total = counts[i, :].sum()
        if row_total > 0:
            pct[i, :] = 100.0 * counts[i, :] / row_total

    pct_df = pd.DataFrame(pct, index=DOMAIN_ORDER, columns=stages)
    counts_df = pd.DataFrame(counts.astype(int), index=DOMAIN_ORDER, columns=stages)
    return pct_df, counts_df


def _course_level_stage_share_full(course_df: pd.DataFrame,
                                   college_filter: str | None = None):
    """Return (pct_df, counts_df, totals_series).

    The cell denominator here is the per-domain TOTAL course count
    (totals_series), NOT the row sum: a course flagged as both Pre_Medical and
    Medical is counted in both the Pre and Med cells, so a row can sum just
    above 100%. counts_df holds the raw per-cell numerators. Returning both
    lets annotations show "%(numerator/domain total)" with no new counting.
    """
    merged = _load_year_merged(course_df)
    if college_filter is not None:
        merged = merged[merged["College_EN"] == college_filter].copy()

    stages = ["Pre", "Med"]
    counts = np.zeros((len(DOMAIN_ORDER), len(stages)))
    domain_totals = np.zeros(len(DOMAIN_ORDER))
    for i, dom in enumerate(DOMAIN_ORDER):
        col = DOMAIN_CSVCOL[dom]
        sub = merged[merged[col] == 1]
        domain_totals[i] = len(sub)
        for _, row in sub.iterrows():
            if row["Pre_Medical"] == 1:
                counts[i, 0] += 1
            if row["Medical"] == 1:
                counts[i, 1] += 1

    pct = np.full_like(counts, np.nan, dtype=float)
    for i in range(counts.shape[0]):
        if domain_totals[i] > 0:
            pct[i, :] = 100.0 * counts[i, :] / domain_totals[i]

    pct_df = pd.DataFrame(pct, index=DOMAIN_ORDER, columns=stages)
    counts_df = pd.DataFrame(counts.astype(int), index=DOMAIN_ORDER, columns=stages)
    totals_series = pd.Series(domain_totals.astype(int), index=DOMAIN_ORDER)
    return pct_df, counts_df, totals_series


# ============================================================
# Figure 1 (unchanged in structure from v10; only data path differs)
# ============================================================

def figure1(course_df: pd.DataFrame, school_df: pd.DataFrame) -> None:
    """5 panels: (a) domain total credits, (b) credits per school by college,
    (c) adoption rate heatmap, (d) grade-resolved year × domain heatmap,
    (e) stage-collapsed full-sample heatmap.
    """
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.00))
    gs = fig.add_gridspec(
        3, 6, hspace=0.30, wspace=1.30,
        width_ratios=[1, 1, 1, 1, 1, 1],
        height_ratios=[1.0, 1.05, 1.05],
    )

    # ---------------- (a) Domain total credits ----------------
    ax_a = fig.add_subplot(gs[0, :3])
    domain_dist = compute_domain_distribution(course_df, school_df)

    y_pos = np.arange(len(domain_dist))[::-1]
    colors = [DOMAIN_COLORS[d] for d in domain_dist["Domain"]]
    ax_a.barh(y_pos, domain_dist["Total_Credits"], color=colors, height=0.62,
              edgecolor=EDGE_COLOR, linewidth=EDGE_LW)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([DOMAIN_SHORT[d] for d in domain_dist["Domain"]],
                          fontsize=5.8, linespacing=0.95)
    for yi, (cr, pct) in zip(y_pos, zip(domain_dist["Total_Credits"], domain_dist["Pct_Credits"])):
        ax_a.text(cr + 5, yi, f"{cr:.1f} ({pct:.1f}%)",
                  va="center", fontsize=5.5, color="#333333")
    ax_a.set_xlabel("Total credits across 63 schools")
    ax_a.set_xlim(0, domain_dist["Total_Credits"].max() * 1.32)
    _panel_label(ax_a, "a", x=-0.32)

    # ---------------- (b) Credits per school by college (box+strip) ----------------
    ax_b = fig.add_subplot(gs[0, 3:])
    rng = np.random.default_rng(42)
    for i, college in enumerate(COLLEGE_ORDER):
        vals = school_df[school_df["College_EN"] == college]["Total_Credits"].values
        bp = ax_b.boxplot([vals], positions=[i], widths=0.55,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color="black", linewidth=0.9),
                          whiskerprops=dict(linewidth=0.5),
                          capprops=dict(linewidth=0.5),
                          boxprops=dict(linewidth=EDGE_LW, edgecolor=EDGE_COLOR))
        bp["boxes"][0].set_facecolor(COLLEGE_COLORS[college])
        bp["boxes"][0].set_alpha(0.35)
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        ax_b.scatter(np.full(len(vals), i) + jitter, vals,
                     c=COLLEGE_COLORS[college], s=10, alpha=0.75,
                     edgecolors=EDGE_COLOR, linewidths=EDGE_LW, zorder=4)

    ax_b.set_xticks(range(len(COLLEGE_ORDER)))
    n_per_college = {c: int((school_df["College_EN"] == c).sum()) for c in COLLEGE_ORDER}
    ax_b.set_xticklabels(
        [f"{COLLEGE_SHORT[c]}\n(n={n_per_college[c]})" for c in COLLEGE_ORDER],
        fontsize=6.5, linespacing=0.95)
    ax_b.set_ylabel("Total credits per school")
    ax_b.set_ylim(0, school_df["Total_Credits"].max() * 1.18)
    _panel_label(ax_b, "b", x=-0.18)

    # ---------------- (c) Adoption rate heatmap (domain × college) ----------------
    ax_c = fig.add_subplot(gs[1, :3])
    rates = np.zeros((len(DOMAIN_ORDER), len(COLLEGE_ORDER)))
    counts = np.zeros((len(DOMAIN_ORDER), len(COLLEGE_ORDER)))
    for j, college in enumerate(COLLEGE_ORDER):
        sub = school_df[school_df["College_EN"] == college]
        for i, dom in enumerate(DOMAIN_ORDER):
            col = f"has_{DOMAIN_CSVCOL[dom]}"
            n_off = int((sub[col] >= 1).sum())
            rates[i, j] = 100 * n_off / n_per_college[college]
            counts[i, j] = n_off

    base = plt.cm.Blues
    colors_list = [(1, 1, 1)] + [base(t) for t in np.linspace(0.18, 0.95, 254)]
    cmap_blue = LinearSegmentedColormap.from_list("white_blues", colors_list, N=256)

    im_c = ax_c.imshow(rates, cmap=cmap_blue, vmin=0, vmax=100, aspect="equal")
    ax_c.set_xticks(range(len(COLLEGE_ORDER)))
    # Square cells make this 3-column panel narrow; long college names (esp.
    # "Korean medicine") overlap when horizontal, so rotate the x-labels.
    ax_c.set_xticklabels([COLLEGE_SHORT[c] for c in COLLEGE_ORDER], fontsize=6.5,
                         rotation=30, ha="right", rotation_mode="anchor")
    ax_c.set_yticks(range(len(DOMAIN_ORDER)))
    ax_c.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER],
                         fontsize=5.8, linespacing=0.95)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            tcol = "white" if rates[i, j] >= 55 else "black"
            ax_c.text(j, i, f"{rates[i, j]:.1f}%\n({int(counts[i, j])}/{n_per_college[COLLEGE_ORDER[j]]})",
                      ha="center", va="center", fontsize=5.5, color=tcol,
                      linespacing=0.95)
    _draw_cell_grid(ax_c, rates.shape[0], rates.shape[1])
    for s in ax_c.spines.values():
        s.set_visible(False)
    # Colorbar for panel c is created after the canvas draw (below), once the
    # aspect='equal' square image box is known, so it hugs the image instead of
    # the wide slot edge (which would collide with panel d's y-labels).
    _panel_label(ax_c, "c", x=-0.32)

    # ---------------- (d) Grade-resolved year × domain heatmap (6 cols) ----------------
    ax_d = fig.add_subplot(gs[1, 3:])
    stage_share_d, counts_d = _course_level_stage_share_resolved(course_df)

    stage_labels = {
        "Pre-1": "Pre\n1", "Pre-2": "Pre\n2",
        "Med-1": "Med\n1", "Med-2": "Med\n2",
        "Med-3": "Med\n3", "Med-4": "Med\n4",
    }
    cols_d = list(stage_share_d.columns)

    im2 = ax_d.imshow(stage_share_d.values, cmap=cmap_blue,
                      vmin=0, vmax=100, aspect="equal")
    ax_d.set_xticks(range(len(cols_d)))
    ax_d.set_xticklabels([stage_labels[c] for c in cols_d], fontsize=5.6,
                         linespacing=0.95, rotation=0)
    ax_d.set_yticks(range(len(DOMAIN_ORDER)))
    ax_d.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER],
                         fontsize=5.8, linespacing=0.95)
    # Per-row denominator = grade-resolved courses in that domain (81/25/30/17/21).
    den_d = counts_d.sum(axis=1)
    for i in range(stage_share_d.shape[0]):
        for j in range(stage_share_d.shape[1]):
            v = stage_share_d.iloc[i, j]
            if np.isnan(v) or round(v) == 0:
                ax_d.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              facecolor="white", edgecolor="white",
                                              linewidth=0, zorder=2))
                continue
            tcol = "white" if v >= 55 else "black"
            # Match panel (c): ratio(%) over (numerator/denominator), two lines.
            ax_d.text(j, i,
                      f"{v:.1f}%\n({int(counts_d.iloc[i, j])}/{int(den_d.iloc[i])})",
                      ha="center", va="center", fontsize=5.5, color=tcol,
                      linespacing=0.95, zorder=3)
    _draw_cell_grid(ax_d, stage_share_d.shape[0], stage_share_d.shape[1])
    for s in ax_d.spines.values():
        s.set_visible(False)
    cbar2 = fig.colorbar(im2, ax=ax_d, shrink=0.55, aspect=12, pad=0.04)
    cbar2.ax.tick_params(labelsize=5.5, width=0.4)
    cbar2.outline.set_linewidth(0.4)
    cbar2.set_label("Share of domain courses (%)", fontsize=6)
    ax_d.set_xlabel("Curriculum stage (grade-resolved subset)")
    _panel_label(ax_d, "d", x=-0.18)

    # ---------------- (e) Stage-collapsed full-sample heatmap ----------------
    fig.canvas.draw()
    figW, figH = fig.get_size_inches()

    # Panel c colorbar, now that the square image box is known: hug the image.
    pos_c = ax_c.get_position()
    cb_h_c = pos_c.height * 0.55
    cbax_c = fig.add_axes([pos_c.x1 + 0.010,
                           pos_c.y0 + (pos_c.height - cb_h_c) / 2.0,
                           0.011, cb_h_c])
    cbar_c = fig.colorbar(im_c, cax=cbax_c)
    cbar_c.ax.tick_params(labelsize=5.5, width=0.4)
    cbar_c.outline.set_linewidth(0.4)
    cbar_c.set_label("Schools offering (%)", fontsize=6)

    pos_d = ax_d.get_position()
    # Panel d is now aspect='equal', so its drawn image is square-celled and
    # centred inside its allocated box. Derive the TRUE square-cell side (in.)
    # from the limiting dimension, then size panel e so its 2x5 cells are
    # square AND the same physical size as panel d's cells.
    s_cell = min(pos_d.width * figW / 6.0,
                 pos_d.height * figH / len(DOMAIN_ORDER))
    e_width = (2 * s_cell) / figW
    e_height = (len(DOMAIN_ORDER) * s_cell) / figH
    bbox_row2 = gs[2, 0].get_position(fig)
    e_bottom = bbox_row2.y0 + (bbox_row2.height - e_height) * 0.5
    e_left = 0.5 - e_width / 2.0
    ax_e = fig.add_axes([e_left, e_bottom, e_width, e_height])
    cb_pad = 0.005
    cb_w = e_height * 0.55 / 12
    cb_h = e_height * 0.55
    cb_left = e_left + e_width + cb_pad
    cb_bottom = e_bottom + (e_height - cb_h) / 2
    cax_e = fig.add_axes([cb_left, cb_bottom, cb_w, cb_h])
    stage_share_e, counts_e, totals_e = _course_level_stage_share_full(course_df)

    stage_labels_e = {"Pre": "Premed", "Med": "Med"}
    cols_e = list(stage_share_e.columns)

    im3 = ax_e.imshow(stage_share_e.values, cmap=cmap_blue,
                      vmin=0, vmax=100, aspect="equal")
    ax_e.set_xticks(range(len(cols_e)))
    ax_e.set_xticklabels([stage_labels_e[c] for c in cols_e], fontsize=5.6)
    ax_e.set_yticks(range(len(DOMAIN_ORDER)))
    ax_e.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER],
                         fontsize=5.8, linespacing=0.95)
    # Per-row denominator = total courses in that domain (106/33/36/22/23).
    # NOT the row sum: a course tagged both premed and med is counted in both
    # cells, so the Health-informatics row sums just above 100%.
    for i in range(stage_share_e.shape[0]):
        for j in range(stage_share_e.shape[1]):
            v = stage_share_e.iloc[i, j]
            if np.isnan(v) or round(v) == 0:
                ax_e.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              facecolor="white", edgecolor="white",
                                              linewidth=0, zorder=2))
                continue
            tcol = "white" if v >= 55 else "black"
            # Match panel (c): ratio(%) over (numerator/denominator), two lines.
            ax_e.text(j, i,
                      f"{v:.1f}%\n({int(counts_e.iloc[i, j])}/{int(totals_e.iloc[i])})",
                      ha="center", va="center", fontsize=5.5, color=tcol,
                      linespacing=0.95, zorder=3)
    _draw_cell_grid(ax_e, stage_share_e.shape[0], stage_share_e.shape[1])
    for s in ax_e.spines.values():
        s.set_visible(False)
    cbar3 = fig.colorbar(im3, cax=cax_e)
    cbar3.ax.tick_params(labelsize=5.5, width=0.4)
    cbar3.outline.set_linewidth(0.4)
    cbar3.set_label("Share of domain courses (%)", fontsize=6)
    ax_e.set_xlabel("Curriculum stage")
    _panel_label(ax_e, "e", x=-0.55)

    _save(fig, "Figure_1")


# ============================================================
# Figure 2 — mandatory share + institutionalization map (2 panels)
# ============================================================

def figure2(course_df: pd.DataFrame, school_df: pd.DataFrame,
            summary: dict) -> None:
    """2 panels: (a) mandatory share by domain, (b) institutionalization map.

    The post-submission revision retired the author-driven 'AI-core' construct,
    so the former panel (c) — AI-core breadth distribution by profession — is
    removed. Panels (a) and (b) are unchanged in data and values; the figure is
    re-laid-out from three columns to two.
    """
    # Width set to 1.5-column so two square-ish panels read comfortably; height
    # follows the single-row content.
    fig = plt.figure(figsize=(ONE_HALF_COL, ONE_HALF_COL * 0.50))
    gs = fig.add_gridspec(
        1, 2,
        wspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---------------- (a) Mandatory share by domain ----------------
    mandatory_df = compute_mandatory_by_domain(course_df)
    y_pos = np.arange(len(mandatory_df))[::-1]
    colors = [DOMAIN_COLORS[d] for d in mandatory_df["Domain"]]
    ax_a.barh(y_pos, mandatory_df["Pct_Mandatory"], color=colors, height=0.62,
              edgecolor=EDGE_COLOR, linewidth=EDGE_LW)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([DOMAIN_SHORT[d] for d in mandatory_df["Domain"]],
                         fontsize=5.8, linespacing=0.95)
    ax_a.set_xlabel("Mandatory courses (%)")
    ax_a.set_xlim(0, 128)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.axvline(50, color=GRID_COLOR, lw=0.5, ls="--", zorder=0)
    for yi, (pct, nm, nt) in zip(y_pos, zip(
            mandatory_df["Pct_Mandatory"],
            mandatory_df["N_Mandatory"],
            mandatory_df["N_Courses"])):
        ax_a.text(pct + 2, yi, f"{pct:.1f}% ({int(nm)}/{int(nt)})",
                  va="center", fontsize=5.5, color="#333333")
    _panel_label(ax_a, "a", x=-0.42)

    # ---------------- (b) Institutionalization map ----------------
    domain_dist = compute_domain_distribution(course_df, school_df)
    merge_b = mandatory_df.merge(domain_dist[["Domain", "Pct_Schools", "Pct_Credits"]],
                                 on="Domain")
    merge_b = merge_b.set_index("Domain").loc[DOMAIN_ORDER].reset_index()

    K_AREA = 7.71
    sizes = K_AREA * merge_b["Pct_Credits"].to_numpy()
    colors_b = [DOMAIN_COLORS[d] for d in merge_b["Domain"]]
    ax_b.scatter(merge_b["Pct_Schools"], merge_b["Pct_Mandatory"],
                 s=sizes, c=colors_b, edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
                 alpha=0.92, zorder=3)

    label_offsets = {
        "Quantitative Foundations": (-8.0, 0.0,  "right",  "center"),
        "AI & Machine Learning":    ( 4.0, 0.0,  "left",   "center"),
        "Data Science":             ( 0.0, -5.5, "center", "top"),
        "Health Informatics":       ( 4.0, 0.0,  "left",   "center"),
        "Clinical AI Application":  (-5.5, 5.0,  "right",  "bottom"),
    }
    for _, row in merge_b.iterrows():
        dx, dy, ha, va = label_offsets[row["Domain"]]
        ax_b.text(row["Pct_Schools"] + dx, row["Pct_Mandatory"] + dy,
                  DOMAIN_ABBR[row["Domain"]],
                  ha=ha, va=va, fontsize=6,
                  fontweight="bold" if row["Domain"] == "Quantitative Foundations" else "normal",
                  color="#222222")

    ax_b.set_xlim(0, 115)
    ax_b.set_ylim(0, 100)
    ax_b.set_xlabel("Schools offering domain (% of n=63)")
    ax_b.set_ylabel("Mandatory courses (%)")
    ax_b.axhline(50, color=GRID_COLOR, lw=0.5, ls="--", zorder=0)
    ax_b.axvline(50, color=GRID_COLOR, lw=0.5, ls="--", zorder=0)

    legend_pcts = [20, 40, 60]
    handles_b = []
    for p in legend_pcts:
        s = K_AREA * p
        handles_b.append(ax_b.scatter([], [], s=s, c="#BBBBBB",
                                      edgecolor=EDGE_COLOR, linewidth=EDGE_LW))
    ax_b.legend(handles_b, [f"{p}%" for p in legend_pcts],
                title="Share of total\nAI/DS credits",
                loc="center right", bbox_to_anchor=(0.99, 0.30),
                frameon=False,
                fontsize=5.5, title_fontsize=5.5,
                labelspacing=0.7, borderpad=0.3,
                handletextpad=1.6)
    _panel_label(ax_b, "b", x=-0.22)

    _save(fig, "Figure_2")


# ============================================================
# Figure 3 — clinical vs research track distribution (3 panels)
# ============================================================

def figure3(school_df: pd.DataFrame, track_metrics: pd.DataFrame,
            track_summary: dict) -> None:
    """3 panels:
      (a) Dot plot of (clinical M2-binary, research M2-binary) by school,
          coloured by profession, with jitter. Equal limits + equal aspect so
          the y=x reference line renders at a true 45 deg (no in-figure label;
          the diagonal is described in the caption).
      (b) Grouped bar plot of MEAN M1 crude credits by profession x track
          (clinical vs research) with 95% percentile-bootstrap CI error bars
          (10,000 resamples, seed=42). Lower bounds stay >= 0 by construction.
      (c) Asymmetry counts bar (Both present / Only clinical / Only research /
          Both absent).
    """
    # Size the figure so panel a's grid slot is SQUARE: then the equal-aspect
    # scatter (set below) fills its slot exactly and panels a/b/c share the
    # same drawn height (no vertical gap, no panel-balance violation).
    fig_w = DOUBLE_COL
    wspace = 0.5
    width_ratios = [1.25, 1.10, 1.00]
    _L, _R, _B, _T = 0.125, 0.9, 0.11, 0.88   # matplotlib default subplot margins
    _n = 3
    _S = (_R - _L) / (1.0 + (_n - 1) * wspace / _n)   # sum of axis-width fractions
    w_a_in = (_S * width_ratios[0] / sum(width_ratios)) * fig_w
    fig_h = w_a_in / (_T - _B)                          # axis height (in) == panel a width (in)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 3, wspace=wspace, width_ratios=width_ratios)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # ---------------- (a) Track dot plot ----------------
    rng = np.random.default_rng(11)
    df_a = track_metrics[["University", "College",
                          "clinical_M2_binary", "research_M2_binary"]].copy()
    # jitter strength
    JITTER = 0.18
    for college in COLLEGE_ORDER:
        sub = df_a[df_a["College"] == college]
        xs = sub["clinical_M2_binary"].astype(float).values
        ys = sub["research_M2_binary"].astype(float).values
        xj = xs + rng.uniform(-JITTER, JITTER, len(xs))
        yj = ys + rng.uniform(-JITTER, JITTER, len(ys))
        ax_a.scatter(xj, yj,
                     c=COLLEGE_COLORS[college],
                     s=22, alpha=0.78, edgecolors=EDGE_COLOR, linewidths=EDGE_LW,
                     label=COLLEGE_SHORT[college], zorder=3)

    # 45 deg reference line y=x (described in the caption; no in-figure label).
    max_val = max(df_a["clinical_M2_binary"].max(),
                  df_a["research_M2_binary"].max())
    lim_hi = max_val + 0.8
    ax_a.plot([-0.5, lim_hi], [-0.5, lim_hi],
              color="#888888", linestyle="--", linewidth=0.5, zorder=1)
    # Equal limits + equal aspect so the y=x diagonal renders at a true 45 deg.
    ax_a.set_xlim(-0.5, lim_hi)
    ax_a.set_ylim(-0.5, lim_hi)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Clinical track year-weighted credits")
    ax_a.set_ylabel("Research track year-weighted credits")
    ax_a.legend(loc="upper right", frameon=False, fontsize=5.5,
                handlelength=0.8, handletextpad=0.4)
    _panel_label(ax_a, "a", x=-0.30, y=1.06)

    # ---------------- (b) Mean M1 crude credits by profession x track (bar + 95% bootstrap CI) ----------------
    BAR_TRACKS = [("clinical", "clinical_credits_crude"),
                  ("research", "research_credits_crude")]
    bar_w = 0.34
    rng_boot = np.random.default_rng(42)
    upper_max = 0.0
    for ci, college in enumerate(COLLEGE_ORDER):
        sub = track_metrics[track_metrics["College"] == college]
        for ti, (track, col) in enumerate(BAR_TRACKS):
            pos = ci + (ti - 0.5) * bar_w
            vals = sub[col].astype(float).values
            mean, lo, hi = _bootstrap_ci_mean(vals, rng_boot)
            # asymmetric error lengths; both >= 0 (lower bound bounded by 0)
            err_lo = max(0.0, mean - lo)
            err_hi = max(0.0, hi - mean)
            ax_b.bar(pos, mean, width=bar_w * 0.85,
                     color=TRACK_COLORS[track], alpha=0.55,
                     edgecolor=EDGE_COLOR, linewidth=EDGE_LW, zorder=2)
            ax_b.errorbar(pos, mean, yerr=[[err_lo], [err_hi]],
                          fmt="none", ecolor="#333333", elinewidth=0.7,
                          capsize=2.0, capthick=0.7, zorder=4)
            upper_max = max(upper_max, hi)

    ax_b.set_xticks(np.arange(len(COLLEGE_ORDER)))
    n_per_college = {c: int((track_metrics["College"] == c).sum())
                     for c in COLLEGE_ORDER}
    ax_b.set_xticklabels(
        [f"{COLLEGE_SHORT[c]}\n(n={n_per_college[c]})" for c in COLLEGE_ORDER],
        fontsize=6.0, linespacing=0.95)
    ax_b.set_ylabel("Mean credits per school (unweighted)")
    # Quantitative axis starts at 0 (bootstrap CI lower bounds stay >= 0).
    ax_b.set_ylim(0, max(upper_max * 1.15, 1.0))

    handles_b = [
        Patch(facecolor=TRACK_COLORS["clinical"], alpha=0.55,
              edgecolor=EDGE_COLOR, linewidth=EDGE_LW, label="Clinical track"),
        Patch(facecolor=TRACK_COLORS["research"], alpha=0.55,
              edgecolor=EDGE_COLOR, linewidth=EDGE_LW, label="Research track"),
    ]
    ax_b.legend(handles=handles_b,
                loc="upper center", bbox_to_anchor=(0.5, -0.20),
                ncol=2, frameon=False, fontsize=5.5,
                handletextpad=0.4, columnspacing=1.0)
    _panel_label(ax_b, "b", x=-0.16)

    # ---------------- (c) Asymmetry counts bar ----------------
    asym = track_summary["asymmetry_counts"]
    n_total = asym["n_schools_total"]
    categories = [
        ("Both tracks\npresent",  asym["both_tracks_present"],  "#5D6D7E"),
        ("Only clinical",          asym["only_clinical"],         TRACK_COLORS["clinical"]),
        ("Only research",          asym["only_research"],         TRACK_COLORS["research"]),
        ("Both tracks\nabsent",    asym["both_tracks_absent"],   "#BBBBBB"),
    ]
    y_pos = np.arange(len(categories))[::-1]
    counts = [c[1] for c in categories]
    colors_c = [c[2] for c in categories]
    labels_c = [c[0] for c in categories]
    ax_c.barh(y_pos, counts, height=0.62,
              color=colors_c, edgecolor=EDGE_COLOR, linewidth=EDGE_LW)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels_c, fontsize=6.0, linespacing=0.95)
    for yi, cnt in zip(y_pos, counts):
        pct = 100.0 * cnt / n_total
        ax_c.text(cnt + max(counts) * 0.02, yi, f"{cnt} ({pct:.1f}%)",
                  va="center", fontsize=5.8, color="#333333")
    ax_c.set_xlabel("Schools (n=63)")
    ax_c.set_xlim(0, max(counts) * 1.32)
    _panel_label(ax_c, "c", x=-0.30)

    _save(fig, "Figure_3")


# ============================================================
# Figure 4 — institutional-characteristic axes (2-panel forest)
# ============================================================

# Institutional axes, top-to-bottom on the forest y-axis. Same order in both
# panels so the rows read straight across. Reference category for the two
# profession dummies is medical schools.
INST_AXIS_ORDER = ["Is_Public", "Capital_Area", "Is_Dentistry",
                   "Is_KoreanMedicine", "Quota_scaled"]
INST_AXIS_LABELS = {
    "Is_Public":         "Public (vs private)",
    "Capital_Area":      "Capital area (vs non-capital)",
    "Is_Dentistry":      "Dental (vs medical)",
    "Is_KoreanMedicine": "Korean medicine (vs medical)",
    "Quota_scaled":      "Admission quota (per SD)",
}
# Neutral dark for the single OLS series (a); not a category colour.
OLS_POINT_COLOR = "#333333"


def _fmt_p(p: float) -> str:
    """npj-style P-value string: italic *P*, no leading zero.

    Precision rule (reproduces the two worked examples P=.036 and P=.09):
      * P < .001  -> "P<.001"  (none in this dataset, kept for safety)
      * P < .10   -> three decimals with trailing zeros stripped
                     (0.0359 -> .036, 0.0904 -> .09) so estimates sitting near
                     the .05 boundary are not flattened to two decimals.
      * P >= .10  -> two decimals (.22, .14, .88, .92, .72, ...).
    Significant and non-significant values use the identical format (no
    emphasis), per the brief. The literal "P" is italicised by the caller via
    a math-text "$P$" so this helper returns only the numeric tail + relation.
    """
    if p < 0.001:
        return "<.001"
    if p < 0.10:
        # 3 decimals, then strip only TRAILING zeros: 0.0359 -> .036,
        # 0.0904 -> .090 -> .09.
        s = f"{p:.3f}".rstrip("0")
    else:
        # 2 decimals, trailing zeros KEPT: 0.3011 -> .30, 0.2543 -> .25.
        s = f"{p:.2f}"
    s = s.lstrip("0")            # strip the single leading zero -> .036 / .30
    return "=" + s


def figure4(ols_total_credits: pd.DataFrame,
            track_presence: pd.DataFrame) -> None:
    """2-panel forest figure of institutional-characteristic axes.

    (a) Multivariable OLS coefficients (beta, 95% CI) for per-school total
        credits. One point + horizontal CI per axis; vertical dashed reference
        at 0 (no effect on the additive credit scale).
    (b) Multivariable logistic odds ratios (OR, 95% CI) for holding each career
        track (clinical = blue, research = red, matching Fig 3). Two dodged
        points per axis; log x-axis; vertical dashed reference at OR = 1.

    Both significant and non-significant terms are drawn identically (same
    marker size, full opacity, no significance-based colour/alpha) so the
    reader sees every estimate on equal footing. Visualisation only: every
    value is read straight from the CSVs; nothing is recomputed here.
    """
    # ---- panel (a) data: OLS beta + 95% CI -------------------------------
    ols = ols_total_credits.set_index("variable")

    # ---- panel (b) data: logistic OR + 95% CI per track ------------------
    lg = track_presence[track_presence["variable"].isin(INST_AXIS_ORDER)].copy()

    n_axes = len(INST_AXIS_ORDER)
    # Top-to-bottom: row 0 (Public) at the top.
    y_base = np.arange(n_axes)[::-1].astype(float)

    # Height raised from the original 0.40 to 0.50 of the double-column width to
    # open vertical room for the per-estimate P-value labels: panel (b) stacks a
    # clinical label above and a research label below each dodged pair, and the
    # taller panel keeps the converging inter-row labels from touching while
    # both panels stay the same height (no panel-balance violation).
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.50))
    # Panel (a) is wider to host the long left y-axis labels (shared across the
    # figure); panel (b) need not repeat them.
    gs = fig.add_gridspec(1, 2, wspace=0.10, width_ratios=[1.32, 1.00])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    CAP = 0.10           # half-height of the CI end-cap whiskers
    POINT_S = 26         # marker area (pt^2), identical for every estimate
    CI_LW = 0.9

    # ============ (a) OLS beta (total credits) ============
    # P-value text sits a fixed distance ABOVE each point, centred on beta, so
    # it never overlaps the horizontal CI bar, the end-caps or the y-axis
    # labels (panel a has one estimate per row, rows 1.0 apart).
    P_DY_A = 0.26        # vertical offset of the P label above the point (data y)
    P_FS = 5.2           # P-value font size (pt)
    ax_a.axvline(0.0, color="#888888", linestyle="--", linewidth=0.5, zorder=1)
    for i, ax_name in enumerate(INST_AXIS_ORDER):
        r = ols.loc[ax_name]
        beta = float(r["beta"])
        lo = float(r["CI_low"])
        hi = float(r["CI_high"])
        p = float(r["P"])
        y = y_base[i]
        ax_a.plot([lo, hi], [y, y], color=OLS_POINT_COLOR, lw=CI_LW, zorder=3)
        ax_a.plot([lo, lo], [y - CAP, y + CAP], color=OLS_POINT_COLOR,
                  lw=CI_LW, zorder=3)
        ax_a.plot([hi, hi], [y - CAP, y + CAP], color=OLS_POINT_COLOR,
                  lw=CI_LW, zorder=3)
        ax_a.scatter([beta], [y], s=POINT_S, c=OLS_POINT_COLOR,
                     edgecolor=EDGE_COLOR, linewidth=EDGE_LW, zorder=4)
        ax_a.text(beta, y + P_DY_A, f"$P${_fmt_p(p)}",
                  ha="center", va="bottom", fontsize=P_FS,
                  color=OLS_POINT_COLOR, zorder=5)

    # Symmetric x-limits around 0 with headroom past the widest CI.
    a_lo = float(ols["CI_low"].min())
    a_hi = float(ols["CI_high"].max())
    a_span = max(abs(a_lo), abs(a_hi)) * 1.12
    ax_a.set_xlim(-a_span, a_span)
    ax_a.set_ylim(-0.6, n_axes - 0.4)
    ax_a.set_yticks(y_base)
    ax_a.set_yticklabels([INST_AXIS_LABELS[a] for a in INST_AXIS_ORDER],
                         fontsize=6.5)
    ax_a.set_xlabel("Adjusted β, total credits per school (95% CI)")
    ax_a.set_title("Total credits", fontsize=7.5, pad=3)
    ax_a.grid(axis="x", which="major", color="#EEEEEE", linewidth=0.4, zorder=0)
    _panel_label(ax_a, "a", x=-0.62, y=1.10)

    # ============ (b) Logistic OR (track holding) ============
    OFFSET = 0.18
    track_specs = [
        ("clinical", +OFFSET, TRACK_COLORS["clinical"], "Clinical track"),
        ("research", -OFFSET, TRACK_COLORS["research"], "Research track"),
    ]
    ax_b.axvline(1.0, color="#888888", linestyle="--", linewidth=0.5, zorder=1)
    # Each track's P-value is drawn in the track colour, vertically offset AWAY
    # from the row centre: clinical (upper dodge) gets its label ABOVE its point,
    # research (lower dodge) BELOW its point. This keeps the two labels ~0.7 data
    # units apart so they never overlap regardless of where the markers fall on
    # the log x-axis, and the colour ties each P to its dodged estimate.
    P_DY_B = 0.12        # P label offset from the dodged point (data y), per track
    for track_name, off, color, _label in track_specs:
        sub = lg[lg["track"] == track_name].set_index("variable")
        p_va = "bottom" if off > 0 else "top"          # clinical above, research below
        p_dy = P_DY_B if off > 0 else -P_DY_B
        for i, ax_name in enumerate(INST_AXIS_ORDER):
            r = sub.loc[ax_name]
            or_val = float(r["OR"])
            lo = float(r["CI_low"])
            hi = float(r["CI_high"])
            p = float(r["P"])
            y = y_base[i] + off
            ax_b.plot([lo, hi], [y, y], color=color, lw=CI_LW, zorder=3)
            ax_b.plot([lo, lo], [y - 0.08, y + 0.08], color=color,
                      lw=CI_LW, zorder=3)
            ax_b.plot([hi, hi], [y - 0.08, y + 0.08], color=color,
                      lw=CI_LW, zorder=3)
            ax_b.scatter([or_val], [y], s=POINT_S, c=color,
                         edgecolor=EDGE_COLOR, linewidth=EDGE_LW, zorder=4)
            ax_b.text(or_val, y + p_dy, f"$P${_fmt_p(p)}",
                      ha="center", va=p_va, fontsize=P_FS,
                      color=color, zorder=5)

    ax_b.set_xscale("log")
    # Range chosen so the widest CI (Capital-area research, up to 10.6) is held
    # without truncation; log scale absorbs the asymmetry around OR = 1.
    ax_b.set_xlim(0.13, 14)
    xticks = [0.2, 0.5, 1, 2, 5, 10]
    ax_b.set_xticks(xticks)
    ax_b.set_xticklabels([str(x) for x in xticks], fontsize=6)
    ax_b.set_ylim(-0.6, n_axes - 0.4)
    ax_b.set_yticks(y_base)
    ax_b.set_yticklabels([])          # rows already labelled on panel (a)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xlabel("Adjusted odds ratio, track holding (95% CI), log scale")
    ax_b.set_title("Track holding", fontsize=7.5, pad=3)
    ax_b.grid(axis="x", which="major", color="#EEEEEE", linewidth=0.4, zorder=0)

    handles_b = [
        plt.Line2D([0], [0], color=TRACK_COLORS["clinical"], lw=CI_LW,
                   marker="o", markersize=5, markeredgecolor=EDGE_COLOR,
                   markeredgewidth=EDGE_LW, label="Clinical track"),
        plt.Line2D([0], [0], color=TRACK_COLORS["research"], lw=CI_LW,
                   marker="o", markersize=5, markeredgecolor=EDGE_COLOR,
                   markeredgewidth=EDGE_LW, label="Research track"),
    ]
    ax_b.legend(handles=handles_b, loc="upper center",
                bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
                fontsize=6, handletextpad=0.5, columnspacing=1.5)
    _panel_label(ax_b, "b", x=-0.06, y=1.10)

    _save(fig, "Figure_4")


# ============================================================
# Supplementary Note 1: Per-profession year × domain placement
# (unchanged from v10; only output filename differs)
# ============================================================

def supplementary_note_1(course_df: pd.DataFrame) -> None:
    """Per-profession decomposition of Fig 1d/1e (6 small-multiple heatmaps).

    Cell labels are kept as integer percentages here (unlike Fig 1c/1d/1e,
    which show "%(numerator/denominator)"). This 2x3 layout makes each cell
    only ~13 pt wide, which cannot hold the ~16 pt "(n/d)" fraction line
    without clipping; the integer % stays legible. The fraction notation is
    therefore reserved for the larger Fig 1 cells.
    """
    base = plt.cm.Blues
    colors_list = [(1, 1, 1)] + [base(t) for t in np.linspace(0.18, 0.95, 254)]
    cmap_blue = LinearSegmentedColormap.from_list("white_blues", colors_list, N=256)

    professions = [
        ("Medicine",        "Medical schools"),
        ("Dentistry",       "Dental schools"),
        ("Korean Medicine", "Korean medicine schools"),
    ]

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))
    gs = fig.add_gridspec(
        2, 3,
        hspace=0.70, wspace=0.55,
        height_ratios=[1.0, 1.0],
    )

    panel_letters_top = ["a", "b", "c"]
    panel_letters_bot = ["d", "e", "f"]

    stage_labels_d = {
        "Pre-1": "Pre\n1", "Pre-2": "Pre\n2",
        "Med-1": "Med\n1", "Med-2": "Med\n2",
        "Med-3": "Med\n3", "Med-4": "Med\n4",
    }
    stage_labels_e = {"Pre": "Premed", "Med": "Med"}

    last_im_top = None
    for j, (prof_key, prof_label) in enumerate(professions):
        ax = fig.add_subplot(gs[0, j])
        # Supp Note 1 keeps integer-only cell labels (see note in supplementary_
        # note_1 docstring / caption): its small-multiple cells are ~13 pt wide,
        # too narrow for the two-line "%(n/d)" string, so counts are unused here.
        share, _ = _course_level_stage_share_resolved(course_df, college_filter=prof_key)
        cols = list(share.columns)
        im = ax.imshow(share.values, cmap=cmap_blue, vmin=0, vmax=100, aspect="equal")
        last_im_top = im
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([stage_labels_d[c] for c in cols], fontsize=5.6,
                           linespacing=0.95, rotation=0)
        ax.set_yticks(range(len(DOMAIN_ORDER)))
        ax.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER],
                           fontsize=5.5, linespacing=0.95)
        for ii in range(share.shape[0]):
            row_vals = share.iloc[ii, :].values
            if np.all(np.isnan(row_vals)):
                for jj in range(share.shape[1]):
                    ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1,
                                                facecolor="#DDDDDD",
                                                edgecolor="white",
                                                linewidth=0, zorder=2))
                ax.text(share.shape[1] / 2 - 0.5, ii, "n=0",
                        ha="center", va="center", fontsize=5.2,
                        color="#666666", zorder=3)
                continue
            for jj in range(share.shape[1]):
                v = share.iloc[ii, jj]
                if np.isnan(v) or round(v) == 0:
                    ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1,
                                                facecolor="white",
                                                edgecolor="white",
                                                linewidth=0, zorder=2))
                    continue
                tcol = "white" if v >= 55 else "black"
                ax.text(jj, ii, f"{v:.0f}", ha="center", va="center",
                        fontsize=5.4, color=tcol, zorder=3)
        _draw_cell_grid(ax, share.shape[0], share.shape[1])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Grade-resolved subset", fontsize=6)
        ax.set_title(prof_label, fontsize=7, pad=4)
        _panel_label(ax, panel_letters_top[j], x=-0.22, y=1.12)

    if last_im_top is not None:
        cbar_top = fig.colorbar(last_im_top, ax=fig.axes[:3], shrink=0.55,
                                aspect=14, pad=0.02)
        cbar_top.ax.tick_params(labelsize=5.5, width=0.4)
        cbar_top.outline.set_linewidth(0.4)
        cbar_top.set_label("Share of domain courses (%)", fontsize=6)

    last_im_bot = None
    bottom_axes = []
    for j, (prof_key, prof_label) in enumerate(professions):
        ax = fig.add_subplot(gs[1, j])
        bottom_axes.append(ax)
        # Supp Note 1 keeps integer-only cell labels (narrow small-multiple
        # cells); the count/total returns are unused here.
        share, _, _ = _course_level_stage_share_full(course_df, college_filter=prof_key)
        cols = list(share.columns)
        im = ax.imshow(share.values, cmap=cmap_blue, vmin=0, vmax=100, aspect="equal")
        last_im_bot = im
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([stage_labels_e[c] for c in cols], fontsize=5.6)
        ax.set_yticks(range(len(DOMAIN_ORDER)))
        ax.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER],
                           fontsize=5.5, linespacing=0.95)
        for ii in range(share.shape[0]):
            row_vals = share.iloc[ii, :].values
            if np.all(np.isnan(row_vals)):
                for jj in range(share.shape[1]):
                    ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1,
                                                facecolor="#DDDDDD",
                                                edgecolor="white",
                                                linewidth=0, zorder=2))
                ax.text(share.shape[1] / 2 - 0.5, ii, "n=0",
                        ha="center", va="center", fontsize=5.2,
                        color="#666666", zorder=3)
                continue
            for jj in range(share.shape[1]):
                v = share.iloc[ii, jj]
                if np.isnan(v) or round(v) == 0:
                    ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1,
                                                facecolor="white",
                                                edgecolor="white",
                                                linewidth=0, zorder=2))
                    continue
                tcol = "white" if v >= 55 else "black"
                ax.text(jj, ii, f"{v:.0f}", ha="center", va="center",
                        fontsize=5.6, color=tcol, zorder=3)
        _draw_cell_grid(ax, share.shape[0], share.shape[1])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Full sample (collapsed)", fontsize=6)
        ax.set_title(prof_label, fontsize=7, pad=4)
        # These 2-column panels are narrow once squared, so the centred title
        # spans wider than the panel; place the label clearly ABOVE the title.
        _panel_label(ax, panel_letters_bot[j], x=-0.30, y=1.30)

    if last_im_bot is not None:
        cbar_bot = fig.colorbar(last_im_bot, ax=bottom_axes, shrink=0.55,
                                aspect=14, pad=0.02)
        cbar_bot.ax.tick_params(labelsize=5.5, width=0.4)
        cbar_bot.outline.set_linewidth(0.4)
        cbar_bot.set_label("Share of domain courses (%)", fontsize=6)

    _save(fig, "Supplementary_Note_1_PerProfession")


# ============================================================
# Main
# ============================================================

def main() -> int:
    print("=" * 70)
    print("Figures (1, 2, 3, 4) + Supplementary Note 1")
    print("=" * 70)

    setup_style()
    course_df = load_classification()
    school_df = load_school_v11()
    summary = load_summary_v11()
    track_metrics = load_track_metrics_v11()
    track_summary = load_track_summary_v11()
    track_presence = load_track_presence_logistic_v11()
    ols_total_credits = load_continuous_ols_total_credits_v11()

    print(f"  Loaded {len(course_df)} courses across {len(school_df)} schools")
    print(f"  Schools per college: " +
          ", ".join(f"{c}={(school_df['College_EN']==c).sum()}" for c in COLLEGE_ORDER))
    print(f"  Capital_Area: " +
          ", ".join(f"{r}={(school_df['Region_Binary']==r).sum()}"
                    for r in ["Capital area", "Non-Capital area"]))
    print(f"  Governance: " +
          ", ".join(f"{g}={(school_df['Governance']==g).sum()}"
                    for g in ["Public", "Private"]))
    print(f"  Track metrics rows: {len(track_metrics)}")
    print(f"  Track presence logistic rows: {len(track_presence)}")

    print("\n  Figure 1: Quantitative foundation-centric curriculum...")
    figure1(course_df, school_df)

    print("\n  Figure 2: AI-specific institutionalization (2-panel)...")
    figure2(course_df, school_df, summary)

    print("\n  Figure 3: Clinical vs research track distribution (3-panel)...")
    figure3(school_df, track_metrics, track_summary)

    print("\n  Figure 4: Institutional-characteristic axes (2-panel forest)...")
    figure4(ols_total_credits, track_presence)

    print("\n  Supplementary Note 1: Per-profession year × domain placement...")
    supplementary_note_1(course_df)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
