# -*- coding: utf-8 -*-
"""
Primary analysis for the national cross-sectional study of AI/data-science
coursework across all 63 accredited Korean medical (n=40), dental (n=11), and
Korean-medicine (n=12) schools.

Measures the AI/DS curriculum along five content domains and reports:
  - domain credit distribution and school-level adoption (denominator n=63)
  - course-level mandatory share by domain (chi-square)
  - Friedman test with pairwise Wilcoxon (Holm) on the five domain credits
  - Kruskal-Wallis on total credits across professions
  - a reference-configuration measurement (D1 + >=2 AI-core domains + >=8
    credits) with Fisher exact tests by governance and region
  - year-weighted career-track metrics (common baseline / clinical / research)
  - BCa bootstrap confidence intervals for domain adoption
  - classification-strategy sensitivity

Inputs (place under data/ per the README):
  data/classification/final_adjudicated_classification.csv
  data/classification/m1_rules.csv
  data/classification/m2_llm.csv
  data/raw/교육과정현황조사 최종본.xlsx
  data/raw/대학정보.xlsx

Outputs: results/statistics/.
"""

import io
import sys
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import odds_ratio

ROOT = Path(__file__).resolve().parent.parent
CLS_PATH = ROOT / 'data' / 'classification' / 'final_adjudicated_classification.csv'
M1_PATH = ROOT / 'data' / 'classification' / 'm1_rules.csv'
M2_PATH = ROOT / 'data' / 'classification' / 'm2_llm.csv'
UNI_INFO_PATH = ROOT / 'data' / 'raw' / '대학정보.xlsx'
RAW_CURR_FILE = ROOT / 'data' / 'raw' / '교육과정현황조사 최종본.xlsx'
OUT = ROOT / 'results' / 'statistics'
OUT.mkdir(parents=True, exist_ok=True)

DOMAINS = ['D1', 'D2', 'D3', 'D4', 'D5']
DOMAIN_LABEL = {
    'D1': 'Quantitative Foundations', 'D2': 'AI and Machine Learning',
    'D3': 'Data Science', 'D4': 'Health Informatics', 'D5': 'Clinical AI Application',
}
AI_CORE = ['D2', 'D3', 'D5']

# Career-track domain mapping
TRACK_DOMAINS = {
    'baseline': ['D1'],
    'clinical': ['D4', 'D5'],
    'research': ['D2', 'D3'],
}
TRACK_LABEL = {
    'baseline': 'Common baseline (D1)',
    'clinical': 'Clinical-application track (D4 + D5)',
    'research': 'Research-development track (D2 + D3)',
}

COLLEGE_KO2EN = {'의대': 'Medicine', '치대': 'Dentistry', '한의대': 'Korean Medicine'}
REGION_KO2EN = {
    '서울경기인천': 'Seoul/Gyeonggi', '강원권': 'Gangwon', '충청권': 'Chungcheong',
    '전라제주권': 'Jeolla/Jeju', '경남권': 'Gyeongnam', '경북권': 'Gyeongbuk',
}
UNI_COL_RENAME = {
    '학교구분': 'College', '학교명': 'University', '지역': 'Region',
    '입학정원': 'Admission_Quota', '국립사립': 'Public_Private',
}
SCHOOL_NAME_MAP = {
    ('단국대학교', 'Dentistry'): ('단국대학교 글로컬캠퍼스', 'Dentistry'),
    ('차의과대학교', 'Medicine'): ('차의과학대학교', 'Medicine'),
}
CAPITAL_REGIONS = {'Seoul/Gyeonggi'}  # Capital area = Seoul + Gyeonggi + Incheon
RNG = np.random.default_rng(42)

# Year-stage weight functions
LATE_STAGES = {'Med-3', 'Med-4'}
YEAR_WEIGHT_BINARY = {  # primary: clinical-proximity binary
    'Pre-1': 0.5, 'Pre-2': 0.5, 'Med-1': 1.0, 'Med-2': 1.0, 'Med-3': 2.0, 'Med-4': 2.0,
}
YEAR_WEIGHT_LINEAR = {  # sensitivity: linear 1..6
    'Pre-1': 1.0, 'Pre-2': 2.0, 'Med-1': 3.0, 'Med-2': 4.0, 'Med-3': 5.0, 'Med-4': 6.0,
}
YEAR_WEIGHT_CLINICAL_ONLY = {  # sensitivity: clinical-only 0/1
    'Pre-1': 0.0, 'Pre-2': 0.0, 'Med-1': 0.0, 'Med-2': 0.0, 'Med-3': 1.0, 'Med-4': 1.0,
}


# ============================================================
# Loaders
# ============================================================

def load_university_info():
    """Institutional metadata for the 63 schools (rename by column header)."""
    df = pd.read_excel(UNI_INFO_PATH)
    df = df.rename(columns={k: v for k, v in UNI_COL_RENAME.items() if k in df.columns})
    df['College'] = df['College'].map(COLLEGE_KO2EN)
    df['Region'] = df['Region'].map(REGION_KO2EN)
    return df


def load_classification(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['CollegeEN'] = df['College'].map(COLLEGE_KO2EN)

    def remap(row):
        key = (row['University'], row['CollegeEN'])
        return SCHOOL_NAME_MAP.get(key, key)

    df[['SchoolUni', 'SchoolColl']] = df.apply(lambda r: pd.Series(remap(r)), axis=1)
    for d in DOMAINS:
        df[d] = df[d].fillna(0).astype(int)
    df['Credits'] = df['Credits'].fillna(0).astype(float)
    mand_map = {'필수': 1, '선택': 0, 1: 1, 0: 0, '1': 1, '0': 0, True: 1, False: 0}
    df['Is_Mandatory'] = df['Is_Mandatory'].map(mand_map).fillna(0).astype(int)
    return df


def load_year_metadata():
    """Parse Year_Raw from the curriculum Excel into (Pre-1, Pre-2, Med-1..Med-4)."""
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
    return raw[["Course_ID", "Year_Stage", "Pre_Medical", "Medical"]]


# ============================================================
# School-level features
# ============================================================

def _config_flags(n_courses, has_D1, aicore_count, total_credits):
    """Measurement partition of a school by AI/DS coverage.

    reference_config : D1 offered AND >=2 AI-core domains AND >=8 total credits
    no_courses       : offers no AI/DS course
    no_aicore        : offers courses but no AI-core domain
    aicore_below     : offers >=1 AI-core domain but does not meet the config
    """
    meets_config = int(bool(has_D1) and aicore_count >= 2 and total_credits >= 8)
    no_courses = int(n_courses == 0)
    no_aicore = int(n_courses > 0 and aicore_count == 0)
    aicore_below = int(aicore_count >= 1 and not meets_config)
    return meets_config, no_courses, no_aicore, aicore_below


def build_school_features(meta, cls):
    rows = []
    for _, m in meta.iterrows():
        uni, coll = m['University'], m['College']
        sub = cls[(cls['SchoolUni'] == uni) & (cls['SchoolColl'] == coll)]
        feat = {
            'University': uni, 'College': coll,
            'Region': m['Region'], 'Public_Private': m['Public_Private'],
            'Admission_Quota': int(m['Admission_Quota']),
            'Capital_Area': int(m['Region'] in CAPITAL_REGIONS),
            'n_courses': len(sub),
            'total_credits': float(sub['Credits'].sum()),
        }
        for d in DOMAINS:
            feat[f'{d}_n'] = int(sub[d].sum())
            feat[f'{d}_credits'] = float(sub.loc[sub[d] == 1, 'Credits'].sum())
            feat[f'has_{d}'] = int(feat[f'{d}_n'] > 0)
        feat['aicore_count'] = sum(feat[f'has_{d}'] for d in AI_CORE)

        sub_aicore = sub[sub[AI_CORE].max(axis=1) == 1]
        feat['has_mandatory_aicore'] = int((sub_aicore['Is_Mandatory'] == 1).any()) if len(sub_aicore) else 0

        mc, _, no_ai, ai_below = _config_flags(
            feat['n_courses'], feat['has_D1'], feat['aicore_count'], feat['total_credits'])
        feat['meets_reference_config'] = mc
        feat['meets_any_aicore'] = int(feat['aicore_count'] >= 1)
        feat['no_aicore_with_courses'] = no_ai
        feat['aicore_below_config'] = ai_below
        rows.append(feat)
    return pd.DataFrame(rows)


# ============================================================
# Track + year-weighted metrics
# ============================================================

def compute_track_metrics(cls_with_year, meta):
    """Per-school, per-track metrics (M1 crude, M2 year-weighted x3, M3 LSE, M4 spiral)."""
    out_rows = []
    for _, m in meta.iterrows():
        uni, coll = m['University'], m['College']
        sub = cls_with_year[(cls_with_year['SchoolUni'] == uni) &
                            (cls_with_year['SchoolColl'] == coll)]
        row = {'University': uni, 'College': coll}
        for tkey, tdoms in TRACK_DOMAINS.items():
            in_track = sub[sub[tdoms].max(axis=1) == 1]
            crude = float(in_track['Credits'].sum())
            row[f'{tkey}_n_courses'] = int(len(in_track))
            row[f'{tkey}_credits_crude'] = crude

            if len(in_track):
                row[f'{tkey}_mandatory_credits'] = float(
                    in_track.loc[in_track['Is_Mandatory'] == 1, 'Credits'].sum())
                row[f'{tkey}_mandatory_share'] = (
                    row[f'{tkey}_mandatory_credits'] / crude if crude > 0 else None)
            else:
                row[f'{tkey}_mandatory_credits'] = 0.0
                row[f'{tkey}_mandatory_share'] = None

            year_known = in_track[in_track['Year_Stage'].notna()]
            row[f'{tkey}_n_year_known'] = int(len(year_known))
            row[f'{tkey}_credits_year_known'] = float(year_known['Credits'].sum())

            for wname, wmap in [('binary', YEAR_WEIGHT_BINARY),
                                ('linear', YEAR_WEIGHT_LINEAR),
                                ('clin_only', YEAR_WEIGHT_CLINICAL_ONLY)]:
                w = sum(wmap[y] * c for y, c in zip(year_known['Year_Stage'], year_known['Credits']))
                row[f'{tkey}_M2_{wname}'] = float(w)

            late = year_known[year_known['Year_Stage'].isin(LATE_STAGES)]
            row[f'{tkey}_late_credits'] = float(late['Credits'].sum())
            row[f'{tkey}_LSE'] = (
                row[f'{tkey}_late_credits'] / row[f'{tkey}_credits_year_known']
                if row[f'{tkey}_credits_year_known'] > 0 else None)

            row[f'{tkey}_spiral'] = int(year_known['Year_Stage'].nunique())

        row['clinical_minus_research_crude'] = row['clinical_credits_crude'] - row['research_credits_crude']
        row['clinical_minus_research_M2b'] = row['clinical_M2_binary'] - row['research_M2_binary']
        row['both_tracks_absent'] = int(row['clinical_n_courses'] == 0 and row['research_n_courses'] == 0)
        row['only_clinical'] = int(row['clinical_n_courses'] > 0 and row['research_n_courses'] == 0)
        row['only_research'] = int(row['clinical_n_courses'] == 0 and row['research_n_courses'] > 0)
        row['both_tracks_present'] = int(row['clinical_n_courses'] > 0 and row['research_n_courses'] > 0)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def summarize_track_metrics(track_df):
    summary = {}
    for tkey in TRACK_DOMAINS:
        s = {}
        for col in ['n_courses', 'credits_crude', 'M2_binary', 'M2_linear', 'M2_clin_only',
                    'late_credits', 'spiral']:
            v = track_df[f'{tkey}_{col}'].astype(float)
            s[col] = {
                'mean': float(v.mean()),
                'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                'median': float(v.median()),
                'q25': float(v.quantile(0.25)),
                'q75': float(v.quantile(0.75)),
                'min': float(v.min()),
                'max': float(v.max()),
                'n_zero': int((v == 0).sum()),
            }
        lse = track_df[f'{tkey}_LSE'].dropna().astype(float)
        s['LSE'] = {
            'n_defined': int(len(lse)),
            'mean': float(lse.mean()) if len(lse) else None,
            'median': float(lse.median()) if len(lse) else None,
            'sd': float(lse.std(ddof=1)) if len(lse) > 1 else 0.0,
        }
        summary[tkey] = s

    summary['asymmetry_counts'] = {
        'both_tracks_present': int(track_df['both_tracks_present'].sum()),
        'only_clinical': int(track_df['only_clinical'].sum()),
        'only_research': int(track_df['only_research'].sum()),
        'both_tracks_absent': int(track_df['both_tracks_absent'].sum()),
        'n_schools_total': int(len(track_df)),
    }

    # Clinical presence x research presence 2x2 (Fisher exact + conditional MLE OR):
    # tests specialization (OR<1) vs co-accumulation (OR>1).
    both = int(track_df['both_tracks_present'].sum())
    only_cli = int(track_df['only_clinical'].sum())
    only_res = int(track_df['only_research'].sum())
    neither = int(track_df['both_tracks_absent'].sum())
    co_table = [[both, only_cli], [only_res, neither]]
    co_or, co_ci, co_p = fisher_or_ci(co_table)
    summary['track_cooccurrence_fisher'] = {
        'table_rows_clinical_cols_research': co_table,
        'OR_conditional_MLE': float(co_or),
        'CI95': [float(co_ci[0]), float(co_ci[1])],
        'P_two_sided': float(co_p),
    }

    cli = track_df['clinical_credits_crude'].values
    res = track_df['research_credits_crude'].values
    w_crude = stats.wilcoxon(cli, res, zero_method='wilcox', alternative='two-sided', method='approx')
    summary['wilcoxon_clinical_vs_research_crude'] = {'stat': float(w_crude.statistic), 'P': float(w_crude.pvalue)}
    cli2 = track_df['clinical_M2_binary'].values
    res2 = track_df['research_M2_binary'].values
    w_m2 = stats.wilcoxon(cli2, res2, zero_method='wilcox', alternative='two-sided', method='approx')
    summary['wilcoxon_clinical_vs_research_M2binary'] = {'stat': float(w_m2.statistic), 'P': float(w_m2.pvalue)}

    return summary


# ============================================================
# Helpers (Fisher, BCa)
# ============================================================

def fisher_or_ci(table):
    r = odds_ratio(table, kind='conditional')
    ci = r.confidence_interval(0.95)
    return r.statistic, (ci.low, ci.high), stats.fisher_exact(table)[1]


def bca_bootstrap_proportion(values, n_boot=1000, strata=None):
    values = np.asarray(values, dtype=float)
    point = values.mean()
    if strata is None:
        idx_all = [np.arange(len(values))]
    else:
        strata = np.asarray(strata)
        idx_all = [np.where(strata == s)[0] for s in np.unique(strata)]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        picks = []
        for idx in idx_all:
            picks.append(RNG.choice(idx, size=len(idx), replace=True))
        sample = values[np.concatenate(picks)]
        boots[b] = sample.mean()
    z0 = stats.norm.ppf((np.mean(boots < point)).clip(1e-6, 1 - 1e-6))
    jack = np.empty(len(values))
    for i in range(len(values)):
        jack[i] = np.delete(values, i).mean()
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    alpha = 0.025
    z_lo = z0 + (z0 + stats.norm.ppf(alpha)) / (1 - a * (z0 + stats.norm.ppf(alpha)))
    z_hi = z0 + (z0 + stats.norm.ppf(1 - alpha)) / (1 - a * (z0 + stats.norm.ppf(1 - alpha)))
    return (point,
            float(np.quantile(boots, stats.norm.cdf(z_lo))),
            float(np.quantile(boots, stats.norm.cdf(z_hi))))


# ============================================================
# Main
# ============================================================

def run():
    meta = load_university_info()
    assert len(meta) == 63, f"meta has {len(meta)} rows, expected 63"

    cls = load_classification(CLS_PATH)
    year_df = load_year_metadata()
    cls_year = cls.merge(year_df, on='Course_ID', how='left')

    print(f"classification courses: {len(cls)}, with-year known: "
          f"{cls_year['Year_Stage'].notna().sum()} / unique schools: "
          f"{cls[['SchoolUni','SchoolColl']].drop_duplicates().shape[0]}")

    schools = build_school_features(meta, cls)
    schools.to_csv(OUT / 'per_school.csv', index=False, encoding='utf-8-sig')

    summary = {
        'definition': {
            'reference_configuration': 'D1 + >=2 AI-core domains + >=8 credits',
            'ai_core_domains': AI_CORE,
            'track_mapping': {k: v for k, v in TRACK_DOMAINS.items()},
            'year_weight_primary': 'clinical-proximity binary: Pre 0.5, basic-Med 1.0, clinical-Med 2.0',
        },
        'n_schools_total': len(schools),
        'n_schools_by_college': schools['College'].value_counts().to_dict(),
        'n_courses_total': int(cls.shape[0]),
        'n_courses_year_known': int(cls_year['Year_Stage'].notna().sum()),
        'n_unique_titles': int(cls['Course_Name'].nunique()),
        'n_zero_course_schools': int((schools['n_courses'] == 0).sum()),
        'zero_course_schools': (schools[schools['n_courses'] == 0]
                                [['University', 'College', 'Region', 'Public_Private']]
                                .to_dict('records')),
    }

    total_credits = cls['Credits'].sum()
    summary['credit_share_by_domain'] = {
        DOMAIN_LABEL[d]: float(cls.loc[cls[d] == 1, 'Credits'].sum() / total_credits * 100)
        for d in DOMAINS
    }
    summary['adoption_pct_of_63'] = {
        DOMAIN_LABEL[d]: float(schools[f'has_{d}'].sum() / len(schools) * 100) for d in DOMAINS
    }
    summary['mean_credits_all63'] = float(schools['total_credits'].mean())
    summary['sd_credits_all63'] = float(schools['total_credits'].std(ddof=1))
    summary['mean_credits_courseBearing'] = float(
        schools.loc[schools['n_courses'] > 0, 'total_credits'].mean())
    summary['sd_credits_courseBearing'] = float(
        schools.loc[schools['n_courses'] > 0, 'total_credits'].std(ddof=1))

    # Mandatory by domain (course-level)
    mand_by_domain = {}
    for d in DOMAINS:
        sub = cls[cls[d] == 1]
        n_total = len(sub)
        n_mand = int((sub['Is_Mandatory'] == 1).sum())
        mand_by_domain[DOMAIN_LABEL[d]] = {
            'n_courses_in_domain': n_total,
            'n_mandatory': n_mand,
            'pct_mandatory': float(n_mand / n_total * 100) if n_total else None,
        }
    summary['mandatory_by_domain'] = mand_by_domain

    chi_table = np.array([[mand_by_domain[DOMAIN_LABEL[d]]['n_mandatory'],
                           mand_by_domain[DOMAIN_LABEL[d]]['n_courses_in_domain']
                           - mand_by_domain[DOMAIN_LABEL[d]]['n_mandatory']]
                          for d in DOMAINS])
    chi2, pval, dof, _ = stats.chi2_contingency(chi_table)
    summary['mandatory_chi2'] = {
        'chi2': float(chi2), 'df': int(dof), 'P': float(pval),
        'cramer_v': float(np.sqrt(chi2 / (chi_table.sum() * (min(chi_table.shape) - 1)))),
    }

    # Course-level Domain x curriculum-phase (Premed vs Med) contingency.
    phase_table = np.array([
        [int(((cls_year[d] == 1) & (cls_year['Pre_Medical'] == 1)).sum()),
         int(((cls_year[d] == 1) & (cls_year['Medical'] == 1)).sum())]
        for d in DOMAINS
    ])
    tchi2, tp, tdof, _ = stats.chi2_contingency(phase_table)
    summary['timing_chi2'] = {
        'chi2': float(tchi2), 'df': int(tdof), 'P': float(tp),
        'cramer_v': float(np.sqrt(tchi2 / (phase_table.sum() * (min(phase_table.shape) - 1)))),
        'table_rows_domains_cols_premed_med': phase_table.tolist(),
    }

    # Reference-configuration measurement (D1 + >=2 AI-core domains + >=8 credits).
    summary['reference_config'] = {
        'definition': 'D1 + >=2 AI-core domains + >=8 credits',
        'n_meeting': int(schools['meets_reference_config'].sum()),
        'pct_meeting': float(schools['meets_reference_config'].mean() * 100),
        'by_profession': {
            coll: {
                'n_total': int((schools['College'] == coll).sum()),
                'n_meeting': int(((schools['College'] == coll) & (schools['meets_reference_config'] == 1)).sum()),
                'pct_meeting': float(((schools['College'] == coll) & (schools['meets_reference_config'] == 1)).sum()
                                     / max(1, (schools['College'] == coll).sum()) * 100),
            } for coll in sorted(schools['College'].unique())
        },
        'by_governance': {
            gov: {
                'n_total': int((schools['Public_Private'] == gov).sum()),
                'n_meeting': int(((schools['Public_Private'] == gov) & (schools['meets_reference_config'] == 1)).sum()),
                'pct_meeting': float(((schools['Public_Private'] == gov) & (schools['meets_reference_config'] == 1)).sum()
                                     / max(1, (schools['Public_Private'] == gov).sum()) * 100),
            } for gov in sorted(schools['Public_Private'].unique())
        },
        'by_region': {
            ('Capital' if r == 1 else 'Non-Capital'): {
                'n_total': int((schools['Capital_Area'] == r).sum()),
                'n_meeting': int(((schools['Capital_Area'] == r) & (schools['meets_reference_config'] == 1)).sum()),
                'pct_meeting': float(((schools['Capital_Area'] == r) & (schools['meets_reference_config'] == 1)).sum()
                                     / max(1, (schools['Capital_Area'] == r).sum()) * 100),
            } for r in [1, 0]
        },
    }

    # AI-core breadth distribution (schools by count of AI-core domains covered)
    summary['aicore_breadth_distribution'] = {
        f'{k}_domains': int((schools['aicore_count'] == k).sum()) for k in range(4)
    }
    summary['aicore_breadth_by_profession'] = {
        coll: {
            f'{k}_domains': int(((schools['College'] == coll) & (schools['aicore_count'] == k)).sum())
            for k in range(4)
        } for coll in sorted(schools['College'].unique())
    }
    summary['n_no_aicore_with_courses'] = int(schools['no_aicore_with_courses'].sum())
    summary['n_any_aicore'] = int(schools['meets_any_aicore'].sum())
    summary['n_zero_courses'] = int((schools['n_courses'] == 0).sum())

    def fisher_refconfig_vs_other(group_col, group_a, group_b):
        a_yes = int(((schools[group_col] == group_a) & (schools['meets_reference_config'] == 1)).sum())
        a_no = int(((schools[group_col] == group_a) & (schools['meets_reference_config'] == 0)).sum())
        b_yes = int(((schools[group_col] == group_b) & (schools['meets_reference_config'] == 1)).sum())
        b_no = int(((schools[group_col] == group_b) & (schools['meets_reference_config'] == 0)).sum())
        tab = [[a_yes, a_no], [b_yes, b_no]]
        or_, ci, p = fisher_or_ci(tab)
        return {'table': tab, 'group_a': str(group_a), 'group_b': str(group_b),
                'OR': float(or_),
                'CI_low': float(ci[0]) if np.isfinite(ci[0]) else None,
                'CI_high': float(ci[1]) if np.isfinite(ci[1]) else None,
                'P': float(p)}

    summary['refconfig_fisher_governance'] = fisher_refconfig_vs_other('Public_Private', '국립', '사립')
    summary['refconfig_fisher_region'] = fisher_refconfig_vs_other('Capital_Area', 1, 0)

    kw_groups = [grp['total_credits'].values for _, grp in schools.groupby('College')]
    kw_credits = stats.kruskal(*kw_groups)
    _kw_k = len(kw_groups)
    _kw_N = int(sum(len(g) for g in kw_groups))
    summary['kw_total_credits_by_profession'] = {
        'H': float(kw_credits.statistic), 'P': float(kw_credits.pvalue),
        'k': _kw_k, 'N': _kw_N, 'group_n': [int(len(g)) for g in kw_groups],
        # epsilon-squared effect size for Kruskal-Wallis: (H - k + 1) / (N - k)
        'epsilon_sq': float((kw_credits.statistic - _kw_k + 1) / (_kw_N - _kw_k))}

    # Friedman + pairwise Wilcoxon (Holm) on 5-domain credits
    cred_mat = schools[[f'{d}_credits' for d in DOMAINS]].values
    fried = stats.friedmanchisquare(*[cred_mat[:, i] for i in range(5)])
    summary['friedman_credits'] = {
        'chi2': float(fried.statistic), 'df': 4, 'P': float(fried.pvalue),
        # Kendall's W = chi2 / (n * (k-1)), n = 63 schools, k = 5 domains
        'kendall_W': float(fried.statistic / (len(cred_mat) * 4))}

    pairwise = {}
    pvals = []
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    for i, j in pairs:
        w = stats.wilcoxon(cred_mat[:, i], cred_mat[:, j], zero_method='wilcox',
                           alternative='two-sided', method='approx')
        pvals.append(w.pvalue)
        pairwise[f'{DOMAINS[i]}_vs_{DOMAINS[j]}'] = {'stat': float(w.statistic), 'P_raw': float(w.pvalue)}
    order_idx = np.argsort(pvals)
    holm = [None] * len(pvals)
    m = len(pvals)
    for k, idx in enumerate(order_idx):
        holm[idx] = min(1.0, pvals[idx] * (m - k))
    for k in range(1, len(order_idx)):
        i_prev, i_now = order_idx[k - 1], order_idx[k]
        holm[i_now] = max(holm[i_now], holm[i_prev])
    for (i, j), p_adj in zip(pairs, holm):
        pairwise[f'{DOMAINS[i]}_vs_{DOMAINS[j]}']['P_holm'] = float(p_adj)
    summary['wilcoxon_pairwise'] = pairwise

    # Track metrics (per-school + summary)
    track_df = compute_track_metrics(cls_year, meta)
    track_df.to_csv(OUT / 'track_metrics.csv', index=False, encoding='utf-8-sig')
    track_summary = summarize_track_metrics(track_df)
    with open(OUT / 'track_summary.json', 'w', encoding='utf-8') as f:
        json.dump(track_summary, f, ensure_ascii=False, indent=2, default=str)
    summary['track_summary'] = track_summary

    # Mandatory share stratified (D1 vs AI-core, course-level)
    def school_mandatory_rates(sub_cls):
        d1_courses = sub_cls[sub_cls['D1'] == 1]
        ai_courses = sub_cls[sub_cls[AI_CORE].max(axis=1) == 1]
        return {
            'D1_n': len(d1_courses),
            'D1_mand': int((d1_courses['Is_Mandatory'] == 1).sum()),
            'AI_n': len(ai_courses),
            'AI_mand': int((ai_courses['Is_Mandatory'] == 1).sum()),
            'D1_pct': float((d1_courses['Is_Mandatory'] == 1).mean() * 100) if len(d1_courses) else None,
            'AI_pct': float((ai_courses['Is_Mandatory'] == 1).mean() * 100) if len(ai_courses) else None,
        }

    strat_rows = []
    for axis_col, axis_label in [('College', 'College'), ('Public_Private', 'Governance'),
                                 ('Capital_Area', 'Region')]:
        for level, grp in schools.groupby(axis_col):
            grp_schools = list(zip(grp['University'], grp['College']))
            sub_cls = cls[cls.apply(lambda r: (r['SchoolUni'], r['SchoolColl']) in grp_schools, axis=1)]
            r = school_mandatory_rates(sub_cls)
            r.update({'axis': axis_label, 'level': str(level), 'n_schools': len(grp)})
            if r['D1_n'] and r['AI_n']:
                tab = [[r['D1_mand'], r['D1_n'] - r['D1_mand']],
                       [r['AI_mand'], r['AI_n'] - r['AI_mand']]]
                _or, _ci, _p = fisher_or_ci(tab)
                r['gap_pp'] = (r['D1_pct'] - r['AI_pct']) if r['D1_pct'] is not None and r['AI_pct'] is not None else None
                r['P_fisher'] = float(_p)
            else:
                r['gap_pp'] = None
                r['P_fisher'] = None
            strat_rows.append(r)
    pd.DataFrame(strat_rows).to_csv(OUT / 'mandatory_gap_stratified.csv', index=False, encoding='utf-8-sig')

    # BCa adoption CIs (College-stratified, 1000 resamples, seed 42)
    boot_rows = []
    for d in DOMAINS:
        pt, lo, hi = bca_bootstrap_proportion(schools[f'has_{d}'].values, 1000, schools['College'].values)
        boot_rows.append({'domain': DOMAIN_LABEL[d], 'point': pt * 100, 'CI_low': lo * 100, 'CI_high': hi * 100})
    pd.DataFrame(boot_rows).to_csv(OUT / 'bootstrap_adoption_bca.csv', index=False, encoding='utf-8-sig')

    # Classification-strategy sensitivity: reference-configuration attainment across
    # five course-classification strategies.
    m1 = pd.read_csv(M1_PATH, encoding='utf-8-sig')
    m2 = pd.read_csv(M2_PATH, encoding='utf-8-sig')

    def build_from_alt(alt_df):
        if 'SchoolUni' in alt_df.columns and 'Credits' in alt_df.columns:
            merged = alt_df.copy()
        else:
            cls_meta = cls[['Course_Name', 'Credits', 'Is_Mandatory', 'SchoolUni', 'SchoolColl']]
            merged = alt_df.merge(cls_meta, on='Course_Name', how='left')
        for d in DOMAINS:
            merged[d] = merged[d].fillna(0).astype(int)
        merged['Credits'] = merged['Credits'].fillna(0).astype(float)
        mand_map = {'필수': 1, '선택': 0, 1: 1, 0: 0, '1': 1, '0': 0, True: 1, False: 0}
        if merged['Is_Mandatory'].dtype == object:
            merged['Is_Mandatory'] = merged['Is_Mandatory'].map(mand_map).fillna(0).astype(int)
        else:
            merged['Is_Mandatory'] = merged['Is_Mandatory'].fillna(0).astype(int)
        rows = []
        for _, m_ in meta.iterrows():
            uni, coll = m_['University'], m_['College']
            sub = merged[(merged['SchoolUni'] == uni) & (merged['SchoolColl'] == coll)]
            n_courses = len(sub)
            tc = float(sub['Credits'].sum())
            has = {d: int(sub[d].sum() > 0) for d in DOMAINS}
            ac = sum(has[d] for d in AI_CORE)
            mc, no_c, no_ai, ai_below = _config_flags(n_courses, has['D1'], ac, tc)
            rows.append({'University': uni, 'College': coll,
                         'meets_reference_config': mc, 'no_courses': no_c,
                         'no_aicore_with_courses': no_ai, 'aicore_below_config': ai_below})
        return pd.DataFrame(rows)

    def sens_counts(strategy, df):
        return {'strategy': strategy,
                'n_meets_config': int(df['meets_reference_config'].sum()),
                'pct_meets_config': float(df['meets_reference_config'].mean() * 100),
                'n_no_courses': int(df['no_courses'].sum()),
                'n_no_aicore_with_courses': int(df['no_aicore_with_courses'].sum()),
                'n_aicore_below_config': int(df['aicore_below_config'].sum())}

    sens_rows = []
    primary = schools.copy()
    primary['no_courses'] = (primary['n_courses'] == 0).astype(int)
    sens_rows.append(sens_counts('Consensus (primary)', primary))

    PRIO = ['D5', 'D2', 'D3', 'D4', 'D1']
    cls_prio = cls.copy()
    for i, d in enumerate(PRIO):
        for d2 in PRIO[i + 1:]:
            cls_prio.loc[cls_prio[d] == 1, d2] = 0
    sens_rows.append(sens_counts('Single-domain priority', build_from_alt(cls_prio)))

    n_dom = cls[DOMAINS].sum(axis=1)
    sens_rows.append(sens_counts('Strict single-domain match', build_from_alt(cls[n_dom <= 1].copy())))

    m1n = m1.rename(columns={
        'D1_Quantitative_Foundations': 'D1', 'D2_AI_ML': 'D2',
        'D3_Data_Science': 'D3', 'D4_Health_Informatics': 'D4',
        'D5_Clinical_AI_Application': 'D5'}).copy()
    m1n['CollegeEN'] = m1n['College'].map(COLLEGE_KO2EN)
    m1n[['SchoolUni', 'SchoolColl']] = m1n.apply(
        lambda r: pd.Series(SCHOOL_NAME_MAP.get((r['University'], r['CollegeEN']),
                                                (r['University'], r['CollegeEN']))), axis=1)
    sens_rows.append(sens_counts('Rule-based keyword only', build_from_alt(m1n)))

    cls_meta = cls[['Course_ID', 'Credits', 'Is_Mandatory', 'SchoolUni', 'SchoolColl']]
    m2n = m2.merge(cls_meta, on='Course_ID', how='left')
    sens_rows.append(sens_counts('LLM-assisted only', build_from_alt(m2n)))

    pd.DataFrame(sens_rows).to_csv(OUT / 'sensitivity.csv', index=False, encoding='utf-8-sig')

    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # Headline print
    print("\n========== HEADLINE NUMBERS ==========")
    print(f"N schools: {summary['n_schools_total']} ({summary['n_schools_by_college']})")
    print(f"Zero-course schools: {summary['n_zero_course_schools']}")
    print(f"N courses: {summary['n_courses_total']} (year-known: {summary['n_courses_year_known']}, "
          f"unique titles: {summary['n_unique_titles']})")
    print(f"Mean credits (all 63): {summary['mean_credits_all63']:.2f} ± {summary['sd_credits_all63']:.2f}")
    print("\nDomain school-level adoption (% of 63):")
    for d in DOMAINS:
        print(f"  {DOMAIN_LABEL[d]}: {summary['adoption_pct_of_63'][DOMAIN_LABEL[d]]:.1f}%")
    print("\nMandatory by domain (% of in-domain courses):")
    for d in DOMAINS:
        m_ = mand_by_domain[DOMAIN_LABEL[d]]
        if m_['pct_mandatory'] is not None:
            print(f"  {DOMAIN_LABEL[d]}: {m_['n_mandatory']}/{m_['n_courses_in_domain']} = {m_['pct_mandatory']:.1f}%")
    print(f"Mandatory-by-domain chi-square: {summary['mandatory_chi2']['chi2']:.2f}, "
          f"P={summary['mandatory_chi2']['P']:.3g}")
    rc = summary['reference_config']
    print(f"\nReference configuration (D1 + >=2 AI-core + >=8 credits): "
          f"{rc['n_meeting']}/{summary['n_schools_total']} = {rc['pct_meeting']:.1f}%")
    g = summary['refconfig_fisher_governance']
    print(f"Governance Fisher: OR={g['OR']:.2f} (95% CI {g['CI_low']}-{g['CI_high']}), P={g['P']:.3g}")
    r = summary['refconfig_fisher_region']
    print(f"Region Fisher: OR={r['OR']:.2f} (95% CI {r['CI_low']}-{r['CI_high']}), P={r['P']:.3g}")
    print(f"Friedman credits: chi2={summary['friedman_credits']['chi2']:.2f}, "
          f"W={summary['friedman_credits']['kendall_W']:.3f}, P={summary['friedman_credits']['P']:.3g}")

    print("\n--- Track summary (M1 crude / M2-binary / LSE) ---")
    for tkey in ['baseline', 'clinical', 'research']:
        ts = track_summary[tkey]
        print(f"  {TRACK_LABEL[tkey]}:")
        print(f"    M1 crude credits — mean {ts['credits_crude']['mean']:.2f}, "
              f"median {ts['credits_crude']['median']:.2f}, n_zero {ts['credits_crude']['n_zero']}")
        print(f"    M2-binary year-weighted — mean {ts['M2_binary']['mean']:.2f}, "
              f"median {ts['M2_binary']['median']:.2f}")

    asym = track_summary['asymmetry_counts']
    print(f"\n  Asymmetry counts (of {asym['n_schools_total']} schools): "
          f"both={asym['both_tracks_present']}, only clinical={asym['only_clinical']}, "
          f"only research={asym['only_research']}, both absent={asym['both_tracks_absent']}")

    print("\nClassification-strategy sensitivity (schools meeting reference configuration):")
    for row in sens_rows:
        print(f"  {row['strategy']}: {row['n_meets_config']} ({row['pct_meets_config']:.1f}%)")


if __name__ == '__main__':
    run()
