from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CLASSIF = PROJECT_ROOT / "data" / "classification" / "final_adjudicated_classification.csv"
STATS_DIR = PROJECT_ROOT / "results" / "statistics"
UNI_META_FILE = PROJECT_ROOT / "data" / "raw" / "대학정보.xlsx"
RAW_CURR_FILE = PROJECT_ROOT / "data" / "raw" / "교육과정현황조사 최종본.xlsx"

OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Style
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


# Page sizes
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


# Constants
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

# 4-stage maturity (None at bottom)
MATURITY_ORDER = ["None", "Foundational-Only", "Intermediate", "Advanced"]
MATURITY_LABELS = {
    "None":              "None",
    "Foundational-Only": "Foundational",
    "Intermediate":      "Intermediate",
    "Advanced":          "Advanced",
}
MATURITY_COLORS = {
    "None":              "#CCCCCC",
    "Foundational-Only": "#D65F5F",
    "Intermediate":      "#EE854A",
    "Advanced":          "#4878D0",
}

GOV_COLORS = {"Public": "#4878D0", "Private": "#D4D4D4"}
GRID_COLOR = "#D9D9D9"


# Data loading
def load_classification() -> pd.DataFrame:
    df = pd.read_csv(DATA_CLASSIF)
    df["College_EN"] = df["College"].map(COLLEGE_MAP)
    df["Is_Mandatory_Binary"] = (df["Is_Mandatory"] == "필수").astype(int)
    return df


def load_school() -> pd.DataFrame:
    df = pd.read_csv(STATS_DIR / "per_school.csv", keep_default_na=False)
    for c in ["n_courses", "total_credits",
              "D1_n", "D1_credits", "has_D1",
              "D2_n", "D2_credits", "has_D2",
              "D3_n", "D3_credits", "has_D3",
              "D4_n", "D4_credits", "has_D4",
              "D5_n", "D5_credits", "has_D5",
              "Admission_Quota", "Capital_Area",
              "aicore_count", "has_mandatory_aicore"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    pp_map = {"국립": "Public", "공립": "Public", "사립": "Private"}
    df["Governance"] = df["Public_Private"].map(pp_map)
    df["Region_Binary"] = df["Capital_Area"].map({1: "Capital area",
                                                  0: "Non-Capital area"})
    df["AI_Core_Count"] = df[["has_D2", "has_D3", "has_D5"]].sum(axis=1)
    df["Breadth"] = df[["has_D1", "has_D2", "has_D3", "has_D4", "has_D5"]].sum(axis=1)
    df = df.rename(columns={"total_credits": "Total_Credits",
                            "stage": "Maturity",
                            "College": "College_EN"})
    return df


def load_summary() -> dict:
    with open(STATS_DIR / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


# Course-level derived statistics
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


# Year-stage helpers
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
                                       college_filter: str | None = None) -> pd.DataFrame:
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

    return pd.DataFrame(pct, index=DOMAIN_ORDER, columns=stages)


def _course_level_stage_share_full(course_df: pd.DataFrame,
                                   college_filter: str | None = None) -> pd.DataFrame:
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

    return pd.DataFrame(pct, index=DOMAIN_ORDER, columns=stages)


# Figure 1
def figure1(course_df: pd.DataFrame, school_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 1.12))
    gs = fig.add_gridspec(
        3, 6, hspace=0.62, wspace=1.30,
        width_ratios=[1, 1, 1, 1, 1, 1],
        height_ratios=[1.0, 1.05, 1.05],
    )

    # (a) Domain total credits
    ax_a = fig.add_subplot(gs[0, :3])
    domain_dist = compute_domain_distribution(course_df, school_df)

    y_pos = np.arange(len(domain_dist))[::-1]
    colors = [DOMAIN_COLORS[d] for d in domain_dist["Domain"]]
    ax_a.barh(y_pos, domain_dist["Total_Credits"], color=colors, height=0.62,
              edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([DOMAIN_SHORT[d] for d in domain_dist["Domain"]],
                          fontsize=5.8, linespacing=0.95)
    for yi, (cr, pct) in zip(y_pos, zip(domain_dist["Total_Credits"], domain_dist["Pct_Credits"])):
        ax_a.text(cr + 5, yi, f"{cr:.1f} ({pct:.1f}%)",
                  va="center", fontsize=5.5, color="#333333")
    ax_a.set_xlabel("Total credits across 63 schools")
    ax_a.set_xlim(0, domain_dist["Total_Credits"].max() * 1.32)
    _panel_label(ax_a, "a", x=-0.32)

    # (b) Credits per school by college
    ax_b = fig.add_subplot(gs[0, 3:])
    rng = np.random.default_rng(42)
    for i, college in enumerate(COLLEGE_ORDER):
        vals = school_df[school_df["College_EN"] == college]["Total_Credits"].values
        bp = ax_b.boxplot([vals], positions=[i], widths=0.55,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color="black", linewidth=0.9),
                          whiskerprops=dict(linewidth=0.5),
                          capprops=dict(linewidth=0.5),
                          boxprops=dict(linewidth=0.5))
        bp["boxes"][0].set_facecolor(COLLEGE_COLORS[college])
        bp["boxes"][0].set_alpha(0.35)
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        ax_b.scatter(np.full(len(vals), i) + jitter, vals,
                     c=COLLEGE_COLORS[college], s=10, alpha=0.75,
                     edgecolors="white", linewidths=0.3, zorder=4)

    ax_b.set_xticks(range(len(COLLEGE_ORDER)))
    n_per_college = {c: int((school_df["College_EN"] == c).sum()) for c in COLLEGE_ORDER}
    ax_b.set_xticklabels(
        [f"{COLLEGE_SHORT[c]}\n(n={n_per_college[c]})" for c in COLLEGE_ORDER],
        fontsize=6.5, linespacing=0.95)
    ax_b.set_ylabel("Total credits per school")
    ax_b.set_ylim(0, school_df["Total_Credits"].max() * 1.18)
    _panel_label(ax_b, "b", x=-0.18)

    # (c) Adoption rate heatmap (domain x college)
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

    im = ax_c.imshow(rates, cmap=cmap_blue, vmin=0, vmax=100, aspect="auto")
    ax_c.set_xticks(range(len(COLLEGE_ORDER)))
    ax_c.set_xticklabels([COLLEGE_SHORT[c] for c in COLLEGE_ORDER], fontsize=6.5)
    ax_c.set_yticks(range(len(DOMAIN_ORDER)))
    ax_c.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER],
                         fontsize=5.8, linespacing=0.95)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            tcol = "white" if rates[i, j] >= 55 else "black"
            ax_c.text(j, i, f"{rates[i, j]:.1f}%\n({int(counts[i, j])}/{n_per_college[COLLEGE_ORDER[j]]})",
                      ha="center", va="center", fontsize=5.0, color=tcol,
                      linespacing=0.95)
    for s in ax_c.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.55, aspect=12, pad=0.04)
    cbar.ax.tick_params(labelsize=5.5, width=0.4)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label("Schools offering (%)", fontsize=6)
    _panel_label(ax_c, "c", x=-0.32)

    # (d) Grade-resolved year x domain heatmap
    ax_d = fig.add_subplot(gs[1, 3:])
    stage_share_d = _course_level_stage_share_resolved(course_df)

    stage_labels = {
        "Pre-1": "Pre\n1", "Pre-2": "Pre\n2",
        "Med-1": "Med\n1", "Med-2": "Med\n2",
        "Med-3": "Med\n3", "Med-4": "Med\n4",
    }
    cols_d = list(stage_share_d.columns)

    im2 = ax_d.imshow(stage_share_d.values, cmap=cmap_blue,
                      vmin=0, vmax=100, aspect="auto")
    ax_d.set_xticks(range(len(cols_d)))
    ax_d.set_xticklabels([stage_labels[c] for c in cols_d], fontsize=5.6,
                         linespacing=0.95, rotation=0)
    ax_d.set_yticks(range(len(DOMAIN_ORDER)))
    ax_d.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER],
                         fontsize=5.8, linespacing=0.95)
    for i in range(stage_share_d.shape[0]):
        for j in range(stage_share_d.shape[1]):
            v = stage_share_d.iloc[i, j]
            if np.isnan(v) or round(v) == 0:
                ax_d.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              facecolor="white", edgecolor="white",
                                              linewidth=0, zorder=2))
                continue
            tcol = "white" if v >= 55 else "black"
            ax_d.text(j, i, f"{v:.0f}", ha="center", va="center",
                      fontsize=5.6, color=tcol, zorder=3)
    for s in ax_d.spines.values():
        s.set_visible(False)
    cbar2 = fig.colorbar(im2, ax=ax_d, shrink=0.55, aspect=12, pad=0.04)
    cbar2.ax.tick_params(labelsize=5.5, width=0.4)
    cbar2.outline.set_linewidth(0.4)
    cbar2.set_label("Share of domain courses (%)", fontsize=6)
    ax_d.set_xlabel("Curriculum stage (grade-resolved subset)")
    _panel_label(ax_d, "d", x=-0.18)

    # (e) Stage-collapsed full-sample heatmap
    fig.canvas.draw()
    pos_d = ax_d.get_position()
    cell_w_d = pos_d.width / 6.0
    cell_h_d = pos_d.height / len(DOMAIN_ORDER)
    e_width = 2 * cell_w_d
    e_height = 5 * cell_h_d
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
    stage_share_e = _course_level_stage_share_full(course_df)

    stage_labels_e = {"Pre": "Premed", "Med": "Med"}
    cols_e = list(stage_share_e.columns)

    im3 = ax_e.imshow(stage_share_e.values, cmap=cmap_blue,
                      vmin=0, vmax=100, aspect="auto")
    ax_e.set_xticks(range(len(cols_e)))
    ax_e.set_xticklabels([stage_labels_e[c] for c in cols_e], fontsize=5.6)
    ax_e.set_yticks(range(len(DOMAIN_ORDER)))
    ax_e.set_yticklabels([DOMAIN_SHORT[d] for d in DOMAIN_ORDER],
                         fontsize=5.8, linespacing=0.95)
    for i in range(stage_share_e.shape[0]):
        for j in range(stage_share_e.shape[1]):
            v = stage_share_e.iloc[i, j]
            if np.isnan(v) or round(v) == 0:
                ax_e.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              facecolor="white", edgecolor="white",
                                              linewidth=0, zorder=2))
                continue
            tcol = "white" if v >= 55 else "black"
            ax_e.text(j, i, f"{v:.0f}", ha="center", va="center",
                      fontsize=5.6, color=tcol, zorder=3)
    for s in ax_e.spines.values():
        s.set_visible(False)
    cbar3 = fig.colorbar(im3, cax=cax_e)
    cbar3.ax.tick_params(labelsize=5.5, width=0.4)
    cbar3.outline.set_linewidth(0.4)
    cbar3.set_label("Share of domain courses (%)", fontsize=6)
    ax_e.set_xlabel("Curriculum stage")
    _panel_label(ax_e, "e", x=-0.55)

    _save(fig, "Figure_1")


# Figure 2
def figure2(course_df: pd.DataFrame, school_df: pd.DataFrame,
            summary: dict) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.70))
    gs = fig.add_gridspec(
        2, 6,
        hspace=0.62, wspace=1.30,
        height_ratios=[1.0, 1.0],
    )
    ax_a = fig.add_subplot(gs[0, :3])
    ax_b = fig.add_subplot(gs[0, 3:])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:])

    # (a) Mandatory share by domain
    mandatory_df = compute_mandatory_by_domain(course_df)
    y_pos = np.arange(len(mandatory_df))[::-1]
    colors = [DOMAIN_COLORS[d] for d in mandatory_df["Domain"]]
    ax_a.barh(y_pos, mandatory_df["Pct_Mandatory"], color=colors, height=0.62,
              edgecolor="white", linewidth=0.5)
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
    _panel_label(ax_a, "a", x=-0.32)

    # (b) Institutionalization map
    domain_dist = compute_domain_distribution(course_df, school_df)
    merge_b = mandatory_df.merge(domain_dist[["Domain", "Pct_Schools", "Pct_Credits"]],
                                 on="Domain")
    merge_b = merge_b.set_index("Domain").loc[DOMAIN_ORDER].reset_index()

    K_AREA = 7.71
    sizes = K_AREA * merge_b["Pct_Credits"].to_numpy()
    colors_b = [DOMAIN_COLORS[d] for d in merge_b["Domain"]]
    ax_b.scatter(merge_b["Pct_Schools"], merge_b["Pct_Mandatory"],
                 s=sizes, c=colors_b, edgecolor="black", linewidth=0.5,
                 alpha=0.92, zorder=3)

    # Label offsets tuned to keep crowded mid-left domains (AI/ML, DS, ClinAI)
    # readable: their markers cluster near (x≈27-33, y≈45-48). Three labels are
    # pushed in three different directions (right, down, up-left) to avoid
    # overlap. Quant and HI are isolated and keep their original positions.
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
                                      edgecolor="black", linewidth=0.4))
    ax_b.legend(handles_b, [f"{p}%" for p in legend_pcts],
                title="Share of total\nAI/DS credits",
                loc="center right", bbox_to_anchor=(0.99, 0.30),
                frameon=False,
                fontsize=5.5, title_fontsize=5.5,
                labelspacing=0.7, borderpad=0.3,
                handletextpad=1.6)
    _panel_label(ax_b, "b", x=-0.18)

    # (c) Maturity by college (4-stage stacked bar)
    maturity_cross = pd.crosstab(school_df["College_EN"], school_df["Maturity"])
    for m in MATURITY_ORDER:
        if m not in maturity_cross.columns:
            maturity_cross[m] = 0
    maturity_cross = maturity_cross[MATURITY_ORDER]
    maturity_cross = maturity_cross.reindex(COLLEGE_ORDER)

    n_per_col = maturity_cross.sum(axis=1).values
    pct = maturity_cross.div(maturity_cross.sum(axis=1), axis=0) * 100

    x = np.arange(len(COLLEGE_ORDER))
    bottom = np.zeros(len(COLLEGE_ORDER))
    for m in MATURITY_ORDER:
        vals = pct[m].values
        ax_c.bar(x, vals, bottom=bottom, width=0.55,
                 color=MATURITY_COLORS[m], label=MATURITY_LABELS[m],
                 edgecolor="white", linewidth=0.5)
        bottom += vals

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        [f"{COLLEGE_SHORT[c]}\n(n={int(n_per_col[i])})" for i, c in enumerate(COLLEGE_ORDER)],
        fontsize=5.8, linespacing=0.95)
    ax_c.set_ylabel("Schools (%)")
    ax_c.set_ylim(0, 100)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                ncol=4, frameon=False, fontsize=5.2,
                handletextpad=0.4, columnspacing=0.7)
    _panel_label(ax_c, "c", x=-0.30)

    # (d) Breadth-depth scatter
    rng = np.random.default_rng(42)
    bd = school_df[["University", "College_EN", "Breadth", "Total_Credits", "Maturity"]].copy()
    bd["Depth"] = bd["Total_Credits"]

    for m in MATURITY_ORDER:
        mask = (bd["Maturity"] == m)
        if mask.sum() == 0:
            continue
        xs = bd[mask]["Breadth"].values.astype(float)
        ys = bd[mask]["Depth"].values.astype(float)
        xj = xs + rng.uniform(-0.18, 0.18, len(xs))
        yj = ys + rng.uniform(-0.20, 0.20, len(ys))
        ax_d.scatter(xj, yj, c=MATURITY_COLORS[m],
                     marker="o",
                     s=18, alpha=0.85, edgecolors="white", linewidths=0.4,
                     zorder=3)

    b_med = bd[bd["Maturity"] != "None"]["Breadth"].median()
    d_med = bd[bd["Maturity"] != "None"]["Depth"].median()
    ax_d.axhline(d_med, color=GRID_COLOR, lw=0.5, ls="--", zorder=0)
    ax_d.axvline(b_med, color=GRID_COLOR, lw=0.5, ls="--", zorder=0)

    ax_d.set_xlabel("Breadth (n domains)")
    ax_d.set_ylabel("Total credits")
    ax_d.set_xlim(-0.5, bd["Breadth"].max() + 0.7)
    ax_d.set_xticks(range(0, int(bd["Breadth"].max()) + 1))

    handles_maturity = [
        ax_d.scatter([], [], c=MATURITY_COLORS[m], marker="o",
                     s=18, edgecolors="white",
                     linewidths=0.3,
                     label=MATURITY_LABELS[m])
        for m in MATURITY_ORDER
    ]
    ax_d.legend(handles=handles_maturity, title="Maturity",
                loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=4, frameon=False, fontsize=5.2,
                title_fontsize=5.5,
                handletextpad=0.3, columnspacing=0.7,
                borderpad=0.2)
    _panel_label(ax_d, "d", x=-0.22)

    # (e) Transition resources — both rows report MEAN additional credits for
    # label consistency. Foundational-Only → Intermediate uses the mean credit
    # of an AI-core course in the analyzed cohort (D2|D3|D5, n=54, mean=2.14)
    # as the indicative additional-credit resource for adopting one new
    # AI-core course; Intermediate → Advanced uses the empirical mean gap to
    # 8 credits across Intermediate schools.
    AICORE_COURSE_CREDIT_MEAN = 2.14
    gap = summary["gap_to_next_stage"]
    transitions_stats = [
        {"label": "Foundational\n→ Intermediate",
         "cr_mean": AICORE_COURSE_CREDIT_MEAN,
         "dm_mean": gap["Foundational-Only"]["mean_need_aicore_domains"],
         "mand_mean": gap["Foundational-Only"]["mean_need_mandatory_aicore"],
         "kind": "domain step"},
        {"label": "Intermediate\n→ Advanced",
         "cr_mean": gap["Intermediate"]["mean_need_credits"],
         "dm_mean": gap["Intermediate"]["mean_need_aicore_domains"],
         "mand_mean": gap["Intermediate"]["mean_need_mandatory_aicore"],
         "kind": "credits + mandatory"},
    ]

    y_pos = np.arange(len(transitions_stats))[::-1]
    bar_h = 0.34
    credits_means = [t["cr_mean"] for t in transitions_stats]
    domains_means = [t["dm_mean"] for t in transitions_stats]
    mand_means = [t["mand_mean"] for t in transitions_stats]
    labels = [t["label"] for t in transitions_stats]

    ax_e.barh(y_pos + bar_h / 2, credits_means, bar_h,
              color="#4878D0", label="Additional credits",
              edgecolor="white", linewidth=0.5)
    ax_e.barh(y_pos - bar_h / 2, domains_means, bar_h,
              color="#EE854A", label="Additional AI-core domains",
              edgecolor="white", linewidth=0.5)
    for yi, cr_m, dm_m, mand_m in zip(y_pos, credits_means, domains_means, mand_means):
        if abs(cr_m - round(cr_m)) < 1e-9:
            cr_label = f"{cr_m:.0f}"
        else:
            cr_label = f"{cr_m:.2f}"
        if abs(dm_m - round(dm_m)) < 1e-9:
            dm_label = f"{dm_m:.0f}"
        else:
            dm_label = f"{dm_m:.2f}"
        ax_e.text(cr_m + 0.10, yi + bar_h / 2, cr_label,
                  va="center", fontsize=5.5)
        ax_e.text(dm_m + 0.10, yi - bar_h / 2, dm_label,
                  va="center", fontsize=5.5)

    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels(labels, fontsize=5.5, linespacing=0.95)
    ax_e.set_xlabel("Additional resource (mean)")
    max_val = max(max(credits_means), max(domains_means))
    ax_e.set_xlim(0, max_val * 1.30 + 0.4)
    ax_e.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24),
                ncol=1, frameon=False, fontsize=5.5,
                handletextpad=0.4)
    _panel_label(ax_e, "e", x=-0.32)

    _save(fig, "Figure_2")


# Figure 3
def figure3(course_df: pd.DataFrame, school_df: pd.DataFrame,
            summary: dict) -> None:
    BASE_WIDTH = DOUBLE_COL
    NEW_WIDTH = BASE_WIDTH * (4.1 / 3.0)
    fig = plt.figure(figsize=(NEW_WIDTH, DOUBLE_COL * 0.38))
    gs = fig.add_gridspec(
        1, 3,
        wspace=0.55,
        width_ratios=[2.2, 0.7, 1.2],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # (a) Mandatory % by domain x college
    DOMAIN_FOR_A = DOMAIN_ORDER
    n_dom = len(DOMAIN_FOR_A)
    bar_w = 0.25
    x = np.arange(n_dom)

    for ci, college in enumerate(COLLEGE_ORDER):
        sub = course_df[course_df["College_EN"] == college]
        rates = []
        for dom in DOMAIN_FOR_A:
            col = DOMAIN_CSVCOL[dom]
            dom_courses = sub[sub[col] == 1]
            n = len(dom_courses)
            if n > 0:
                k = int(dom_courses["Is_Mandatory_Binary"].sum())
                p = k / n * 100
            else:
                p = np.nan
            rates.append(p)
        offset = (ci - 1) * bar_w
        rates_arr = np.array(rates, dtype=float)
        ax_a.bar(x + offset, rates_arr, bar_w,
                 color=COLLEGE_COLORS[college], alpha=0.88,
                 edgecolor="white", linewidth=0.4,
                 label=COLLEGE_SHORT[college])

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([DOMAIN_ABBR[d] for d in DOMAIN_FOR_A],
                         fontsize=5.8)
    ax_a.set_ylabel("Mandatory courses (%)")
    ax_a.set_ylim(0, 105)
    ax_a.axhline(50, color=GRID_COLOR, lw=0.5, ls="--", zorder=0)
    ax_a.legend(loc="upper right", frameon=False, fontsize=5.5,
                handlelength=1.0, handletextpad=0.4, ncol=1)
    _panel_label(ax_a, "a", x=-0.12)

    # (b) Maturity by governance (4-stage)
    gov_cross = pd.crosstab(school_df["Governance"], school_df["Maturity"])
    for m in MATURITY_ORDER:
        if m not in gov_cross.columns:
            gov_cross[m] = 0
    gov_cross = gov_cross[MATURITY_ORDER]
    gov_cross = gov_cross.reindex(["Public", "Private"])
    n_per = gov_cross.sum(axis=1).values
    pct_gov = gov_cross.div(gov_cross.sum(axis=1), axis=0) * 100

    x = np.arange(len(gov_cross))
    bottom = np.zeros(len(gov_cross))
    for m in MATURITY_ORDER:
        vals = pct_gov[m].values
        ax_b.bar(x, vals, bottom=bottom, width=0.55,
                 color=MATURITY_COLORS[m], edgecolor="white", linewidth=0.5,
                 label=MATURITY_LABELS[m])
        for i, v in enumerate(vals):
            if v >= 4:
                txt_color = "#333333" if m == "None" else "white"
                ax_b.text(i, bottom[i] + v / 2,
                          f"{v:.1f}%\n(n={int(gov_cross[m].iloc[i])})",
                          ha="center", va="center", fontsize=5.0,
                          color=txt_color, linespacing=0.95)
        bottom += vals

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(
        [f"{g}\n(n={int(n_per[i])})" for i, g in enumerate(gov_cross.index)],
        fontsize=5.8, linespacing=0.95)
    ax_b.set_ylabel("Schools (%)")
    ax_b.set_ylim(0, 100)

    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=4, frameon=False, fontsize=5.2,
                handletextpad=0.3, columnspacing=0.5)
    _panel_label(ax_b, "b", x=-0.25)

    # (c) Stratum-level mandatory gap connecting points
    strat_df = pd.read_csv(STATS_DIR / "mandatory_gap_stratified.csv")

    AXIS_PROF = "#4878D0"
    AXIS_GOV = "#EE854A"
    AXIS_REG = "#6ACC64"

    label_map = {
        "Medicine": "Medical",
        "Dentistry": "Dental",
        "Korean Medicine": "Korean medicine",
        "국립": "Public",
        "사립": "Private",
        "1": "Capital area",
        "0": "Non-Capital area",
    }

    rows = []
    for axis_name, axis_color, axis_key, order_levels in [
        ("Profession", AXIS_PROF, "College",    ["Medicine", "Dentistry", "Korean Medicine"]),
        ("Governance", AXIS_GOV,  "Governance", ["국립", "사립"]),
        ("Region",     AXIS_REG,  "Region",     ["1", "0"]),
    ]:
        df_axis = strat_df[strat_df["axis"] == axis_key].copy()
        df_axis["level"] = df_axis["level"].astype(str)
        for lv in order_levels:
            sub = df_axis[df_axis["level"] == lv]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            rows.append({
                "axis": axis_name,
                "axis_color": axis_color,
                "label": label_map.get(lv, lv),
                "Foundational_Mandatory_Pct": float(r["D1_pct"]),
                "AI_Specific_Mandatory_Pct": float(r["AI_pct"]),
                "Gap_Pct_Points": float(r["gap_pp"]),
                "Fisher_p": float(r["P_fisher"]),
            })
    strat = pd.DataFrame(rows)

    y_positions = np.arange(len(strat))[::-1]
    for yi, (_, row) in zip(y_positions, strat.iterrows()):
        f = row["Foundational_Mandatory_Pct"]
        a = row["AI_Specific_Mandatory_Pct"]
        ax_c.plot([a, f], [yi, yi], color="#666666", lw=0.8, zorder=2)
        ax_c.scatter(a, yi, s=32, color=AI_SPECIFIC_COLOR,
                     edgecolor="black", linewidth=0.4, zorder=4)
        ax_c.scatter(f, yi, s=32, color=FOUNDATIONAL_COLOR,
                     edgecolor="black", linewidth=0.4, zorder=4)
        p = row["Fisher_p"]
        if p < 0.001:
            p_str = "P<.001"
        elif p < 0.01:
            p_str = f"P={p:.3f}".replace("0.", ".")
        else:
            p_str = f"P={p:.2f}".replace("0.", ".")
        ax_c.text(max(a, f) + 4.5, yi,
                  f"{row['Gap_Pct_Points']:.1f} pp ({p_str})",
                  va="center", fontsize=5.5, color="#333333")

    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(strat["label"].tolist(), fontsize=5.8)
    for tick, color in zip(ax_c.get_yticklabels(), strat["axis_color"]):
        tick.set_color(color)

    ax_c.set_xlim(25, 116)
    ax_c.set_xlabel("Mandatory courses (%)")
    ax_c.grid(axis="x", color="#EEEEEE", linewidth=0.4, zorder=0)

    handles_combined = [
        Patch(facecolor=AXIS_PROF, edgecolor="none", label="Profession"),
        Patch(facecolor=AXIS_GOV,  edgecolor="none", label="Governance"),
        Patch(facecolor=AXIS_REG,  edgecolor="none", label="Region"),
        plt.scatter([], [], s=28, color=FOUNDATIONAL_COLOR,
                    edgecolor="black", linewidth=0.4, label="Quant. foundations"),
        plt.scatter([], [], s=28, color=AI_SPECIFIC_COLOR,
                    edgecolor="black", linewidth=0.4, label="AI-specific"),
    ]
    ax_c.legend(handles=handles_combined, loc="upper center",
                bbox_to_anchor=(0.5, -0.16), ncol=5,
                frameon=False, fontsize=5.5,
                handletextpad=0.4, columnspacing=0.8,
                borderpad=0.3)
    _panel_label(ax_c, "c", x=-0.20)

    _save(fig, "Figure_3")


# Supplementary Note 5: per-profession year x domain placement
def supplementary_note_5(course_df: pd.DataFrame) -> None:
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
        share = _course_level_stage_share_resolved(course_df, college_filter=prof_key)
        cols = list(share.columns)
        im = ax.imshow(share.values, cmap=cmap_blue, vmin=0, vmax=100, aspect="auto")
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
        share = _course_level_stage_share_full(course_df, college_filter=prof_key)
        cols = list(share.columns)
        im = ax.imshow(share.values, cmap=cmap_blue, vmin=0, vmax=100, aspect="auto")
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
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Full sample (collapsed)", fontsize=6)
        ax.set_title(prof_label, fontsize=7, pad=4)
        _panel_label(ax, panel_letters_bot[j], x=-0.22, y=1.12)

    if last_im_bot is not None:
        cbar_bot = fig.colorbar(last_im_bot, ax=bottom_axes, shrink=0.55,
                                aspect=14, pad=0.02)
        cbar_bot.ax.tick_params(labelsize=5.5, width=0.4)
        cbar_bot.outline.set_linewidth(0.4)
        cbar_bot.set_label("Share of domain courses (%)", fontsize=6)

    _save(fig, "Supplementary_Note_5_PerProfession")


# Main
def main() -> int:
    print("=" * 70)
    print("Figures (1, 2, 3) + Supplementary Note 5")
    print("=" * 70)

    setup_style()
    course_df = load_classification()
    school_df = load_school()
    summary = load_summary()

    print(f"  Loaded {len(course_df)} courses across {len(school_df)} schools")
    print(f"  Schools per college: " +
          ", ".join(f"{c}={(school_df['College_EN']==c).sum()}" for c in COLLEGE_ORDER))
    print(f"  Maturity counts: " +
          ", ".join(f"{m}={(school_df['Maturity']==m).sum()}" for m in MATURITY_ORDER))
    print(f"  Capital_Area: " +
          ", ".join(f"{r}={(school_df['Region_Binary']==r).sum()}"
                    for r in ["Capital area", "Non-Capital area"]))
    print(f"  Governance: " +
          ", ".join(f"{g}={(school_df['Governance']==g).sum()}"
                    for g in ["Public", "Private"]))

    print("\n  Figure 1: Quantitative foundation-centric curriculum...")
    figure1(course_df, school_df)

    print("\n  Figure 2: AI-specific institutionalization...")
    figure2(course_df, school_df, summary)

    print("\n  Figure 3: Structure of cross-school variation...")
    figure3(course_df, school_df, summary)

    print("\n  Supplementary Note 5: Per-profession year x domain placement...")
    supplementary_note_5(course_df)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
