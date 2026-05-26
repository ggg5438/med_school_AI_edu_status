import io, sys, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import odds_ratio

# Paths
PROJECT = Path(__file__).resolve().parent.parent
CLS_PATH = PROJECT / 'data' / 'classification' / 'final_adjudicated_classification.csv'
M1_PATH = PROJECT / 'data' / 'classification' / 'm1_rules.csv'
M2_PATH = PROJECT / 'data' / 'classification' / 'm2_llm.csv'
UNI_INFO_PATH = PROJECT / 'data' / 'raw' / '대학정보.xlsx'
OUT = PROJECT / 'results' / 'statistics'
OUT.mkdir(parents=True, exist_ok=True)

# Constants
DOMAINS = ['D1', 'D2', 'D3', 'D4', 'D5']
DOMAIN_LABEL = {
    'D1': 'Quantitative Foundations', 'D2': 'AI and Machine Learning',
    'D3': 'Data Science', 'D4': 'Health Informatics', 'D5': 'Clinical AI Application',
}
AI_CORE = ['D2', 'D3', 'D5']
COLLEGE_KO2EN = {'의대': 'Medicine', '치대': 'Dentistry', '한의대': 'Korean Medicine'}
REGION_KO2EN = {
    '서울경기인천': 'Seoul/Gyeonggi',
    '강원권': 'Gangwon',
    '충청권': 'Chungcheong',
    '전라제주권': 'Jeolla/Jeju',
    '경남권': 'Gyeongnam',
    '경북권': 'Gyeongbuk',
}
SCHOOL_NAME_MAP = {
    ('단국대학교', 'Dentistry'): ('단국대학교 글로컬캠퍼스', 'Dentistry'),
    ('차의과대학교', 'Medicine'): ('차의과학대학교', 'Medicine'),
}
CAPITAL_REGIONS = {'Seoul/Gyeonggi'}
RNG = np.random.default_rng(42)


# Data loaders
def load_university_info():
    df = pd.read_excel(UNI_INFO_PATH)
    df.columns = ['College', 'University', 'Region', 'Public_Private', 'Admission_Quota']
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


# School-level feature construction (Option B Advanced, 4-stage maturity)
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
        if feat['n_courses'] == 0:
            feat['stage'] = 'None'
        elif (feat['has_D1'] and feat['aicore_count'] >= 2
              and feat['total_credits'] >= 8 and feat['has_mandatory_aicore']):
            feat['stage'] = 'Advanced'
        elif feat['aicore_count'] >= 1:
            feat['stage'] = 'Intermediate'
        else:
            feat['stage'] = 'Foundational-Only'
        rows.append(feat)
    return pd.DataFrame(rows)


# Statistical helpers
def fisher_or_ci(table):
    a, b = table[0]; c, d = table[1]
    if min(a, b, c, d) == 0:
        r = odds_ratio(table, kind='conditional')
        ci = r.confidence_interval(0.95)
        return r.statistic, (ci.low, ci.high), stats.fisher_exact(table)[1]
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
    return point, float(np.quantile(boots, stats.norm.cdf(z_lo))), float(np.quantile(boots, stats.norm.cdf(z_hi)))


def run():
    meta = load_university_info()
    assert len(meta) == 63, f"meta has {len(meta)} rows, expected 63"

    cls = load_classification(CLS_PATH)
    print(f"classification courses: {len(cls)}, unique schools: "
          f"{cls[['SchoolUni','SchoolColl']].drop_duplicates().shape[0]}")

    schools = build_school_features(meta, cls)
    schools.to_csv(OUT / 'per_school.csv', index=False, encoding='utf-8-sig')

    summary = {'n_schools_total': len(schools),
               'n_schools_by_college': schools['College'].value_counts().to_dict(),
               'n_courses_total': int(cls.shape[0]),
               'n_unique_titles': int(cls['Course_Name'].nunique())}

    summary['n_zero_course_schools'] = int((schools['n_courses'] == 0).sum())
    summary['zero_course_schools'] = (schools[schools['n_courses'] == 0]
                                      [['University', 'College', 'Region', 'Public_Private']]
                                      .to_dict('records'))

    # Domain credit share
    total_credits = cls['Credits'].sum()
    summary['credit_share_by_domain'] = {
        DOMAIN_LABEL[d]: float(cls.loc[cls[d] == 1, 'Credits'].sum() / total_credits * 100)
        for d in DOMAINS
    }

    # School-level adoption rates
    adoption = {DOMAIN_LABEL[d]: float(schools[f'has_{d}'].sum() / len(schools) * 100) for d in DOMAINS}
    summary['adoption_pct_of_63'] = adoption

    # Credits summaries
    summary['mean_credits_all63'] = float(schools['total_credits'].mean())
    summary['sd_credits_all63'] = float(schools['total_credits'].std(ddof=1))
    summary['mean_credits_courseBearing'] = float(
        schools.loc[schools['n_courses'] > 0, 'total_credits'].mean())
    summary['sd_credits_courseBearing'] = float(
        schools.loc[schools['n_courses'] > 0, 'total_credits'].std(ddof=1))

    # Mandatory by domain
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
    chi2, pval, dof, exp = stats.chi2_contingency(chi_table)
    summary['mandatory_chi2'] = {'chi2': float(chi2), 'df': int(dof), 'P': float(pval),
                                 'cramer_v': float(np.sqrt(chi2 / (chi_table.sum() * (min(chi_table.shape) - 1))))}

    # Maturity distribution
    stage_order = ['None', 'Foundational-Only', 'Intermediate', 'Advanced']
    matur_counts = schools['stage'].value_counts().reindex(stage_order, fill_value=0)
    summary['maturity_counts'] = matur_counts.to_dict()
    summary['maturity_pct'] = (matur_counts / len(schools) * 100).round(2).to_dict()

    for axis_col, axis_name in [('College', 'profession'), ('Public_Private', 'governance'),
                                 ('Capital_Area', 'region')]:
        ct = (schools.groupby(axis_col)['stage'].value_counts().unstack(fill_value=0)
              .reindex(columns=stage_order, fill_value=0))
        summary[f'maturity_by_{axis_name}'] = ct.to_dict()

    # Fisher exact for Advanced vs other
    def fisher_advanced_vs_other(group_col, group_a, group_b):
        a_adv = int(((schools[group_col] == group_a) & (schools['stage'] == 'Advanced')).sum())
        a_oth = int(((schools[group_col] == group_a) & (schools['stage'] != 'Advanced')).sum())
        b_adv = int(((schools[group_col] == group_b) & (schools['stage'] == 'Advanced')).sum())
        b_oth = int(((schools[group_col] == group_b) & (schools['stage'] != 'Advanced')).sum())
        tab = [[a_adv, a_oth], [b_adv, b_oth]]
        or_, ci, p = fisher_or_ci(tab)
        return {'table': tab, 'group_a': str(group_a), 'group_b': str(group_b),
                'OR': float(or_), 'CI_low': float(ci[0]) if np.isfinite(ci[0]) else None,
                'CI_high': float(ci[1]) if np.isfinite(ci[1]) else None, 'P': float(p)}

    summary['advanced_fisher_governance'] = fisher_advanced_vs_other('Public_Private', '국립', '사립')
    summary['advanced_fisher_region'] = fisher_advanced_vs_other('Capital_Area', 1, 0)

    # Kruskal-Wallis on credits across profession
    kw_credits = stats.kruskal(*[grp['total_credits'].values for _, grp in schools.groupby('College')])
    summary['kw_total_credits_by_profession'] = {'H': float(kw_credits.statistic), 'P': float(kw_credits.pvalue)}

    # Friedman + pairwise Wilcoxon on domain credits
    cred_mat = schools[[f'{d}_credits' for d in DOMAINS]].values
    fried = stats.friedmanchisquare(*[cred_mat[:, i] for i in range(5)])
    summary['friedman_credits'] = {'chi2': float(fried.statistic), 'df': 4, 'P': float(fried.pvalue)}

    pairwise = {}
    pvals = []
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    for i, j in pairs:
        w = stats.wilcoxon(cred_mat[:, i], cred_mat[:, j], zero_method='wilcox', alternative='two-sided',
                           method='approx')
        pvals.append(w.pvalue)
        pairwise[f'{DOMAINS[i]}_vs_{DOMAINS[j]}'] = {'stat': float(w.statistic), 'P_raw': float(w.pvalue)}
    # Holm-Bonferroni
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

    # Gap to next stage
    def distance_to_next(row):
        if row['stage'] == 'None':
            need_credits = 0; need_aicore = 0
            return {'next': 'Foundational-Only', 'need_credits': 0, 'need_aicore_domains': 0,
                    'need_mandatory_aicore': 0}
        if row['stage'] == 'Foundational-Only':
            return {'next': 'Intermediate', 'need_credits': max(0, 0),
                    'need_aicore_domains': 1, 'need_mandatory_aicore': 0}
        if row['stage'] == 'Intermediate':
            need_credits = max(0, 8 - row['total_credits'])
            need_aicore_domains = max(0, 2 - row['aicore_count'])
            need_mand = 1 - row['has_mandatory_aicore']
            return {'next': 'Advanced', 'need_credits': float(need_credits),
                    'need_aicore_domains': int(need_aicore_domains),
                    'need_mandatory_aicore': int(need_mand)}
        return {'next': None, 'need_credits': None, 'need_aicore_domains': None,
                'need_mandatory_aicore': None}
    gap = schools.apply(lambda r: pd.Series(distance_to_next(r)), axis=1)
    gap_df = pd.concat([schools[['University', 'College', 'stage']], gap], axis=1)
    gap_df.to_csv(OUT / 'gap_analysis.csv', index=False, encoding='utf-8-sig')

    gap_summary = {}
    for stage in ['Foundational-Only', 'Intermediate']:
        sub = gap_df[gap_df['stage'] == stage]
        if len(sub):
            gap_summary[stage] = {
                'n': len(sub),
                'mean_need_credits': float(sub['need_credits'].mean()),
                'sd_need_credits': float(sub['need_credits'].std(ddof=1)) if len(sub) > 1 else 0.0,
                'mean_need_aicore_domains': float(sub['need_aicore_domains'].mean()),
                'mean_need_mandatory_aicore': float(sub['need_mandatory_aicore'].mean()) if 'need_mandatory_aicore' in sub.columns and sub['need_mandatory_aicore'].notna().any() else None,
            }
    summary['gap_to_next_stage'] = gap_summary

    # Stratified mandatory gap (D1 vs AI-core)
    def domain_mandatory_pct(sub_cls, dom):
        ssub = sub_cls[sub_cls[dom] == 1]
        return float((ssub['Is_Mandatory'] == 1).mean() * 100) if len(ssub) else None

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
                r['gap_pp'] = None; r['P_fisher'] = None
            strat_rows.append(r)
    pd.DataFrame(strat_rows).to_csv(OUT / 'mandatory_gap_stratified.csv', index=False, encoding='utf-8-sig')

    # BCa bootstrap CIs for domain adoption
    boot_rows = []
    for d in DOMAINS:
        pt, lo, hi = bca_bootstrap_proportion(schools[f'has_{d}'].values, 1000, schools['College'].values)
        boot_rows.append({'domain': DOMAIN_LABEL[d], 'point': pt * 100, 'CI_low': lo * 100, 'CI_high': hi * 100})
    pd.DataFrame(boot_rows).to_csv(OUT / 'bootstrap_adoption_bca.csv', index=False, encoding='utf-8-sig')

    # Threshold sweep (Option B Advanced credit threshold from 5 to 12)
    sweep = []
    for thr in range(5, 13):
        def stage_thr(row, t=thr):
            if row['n_courses'] == 0:
                return 'None'
            if (row['has_D1'] and row['aicore_count'] >= 2
                    and row['total_credits'] >= t and row['has_mandatory_aicore']):
                return 'Advanced'
            if row['aicore_count'] >= 1:
                return 'Intermediate'
            return 'Foundational-Only'
        s_thr = schools.apply(stage_thr, axis=1)
        adv_n = int((s_thr == 'Advanced').sum())
        a_adv = int(((schools['Public_Private'] == '국립') & (s_thr == 'Advanced')).sum())
        a_oth = int(((schools['Public_Private'] == '국립') & (s_thr != 'Advanced')).sum())
        b_adv = int(((schools['Public_Private'] == '사립') & (s_thr == 'Advanced')).sum())
        b_oth = int(((schools['Public_Private'] == '사립') & (s_thr != 'Advanced')).sum())
        tab = [[a_adv, a_oth], [b_adv, b_oth]]
        try:
            or_, ci, p = fisher_or_ci(tab)
        except Exception:
            or_, ci, p = np.nan, (np.nan, np.nan), np.nan
        sweep.append({'threshold_credits': thr, 'n_advanced': adv_n,
                      'pct_advanced': float(adv_n / len(schools) * 100),
                      'governance_OR': float(or_) if np.isfinite(or_) else None,
                      'governance_CI_low': float(ci[0]) if np.isfinite(ci[0]) else None,
                      'governance_CI_high': float(ci[1]) if np.isfinite(ci[1]) else None,
                      'governance_P': float(p) if np.isfinite(p) else None})
    pd.DataFrame(sweep).to_csv(OUT / 'threshold_sweep.csv', index=False, encoding='utf-8-sig')

    # Classification sensitivity (5 strategies)
    m1 = pd.read_csv(M1_PATH, encoding='utf-8-sig')
    m2 = pd.read_csv(M2_PATH, encoding='utf-8-sig')
    sens_rows = []
    def build_from_alt(alt_df, label):
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
        for _, m in meta.iterrows():
            uni, coll = m['University'], m['College']
            sub = merged[(merged['SchoolUni'] == uni) & (merged['SchoolColl'] == coll)]
            n_courses = len(sub)
            tc = float(sub['Credits'].sum())
            has = {d: int(sub[d].sum() > 0) for d in DOMAINS}
            ac = sum(has[d] for d in AI_CORE)
            sub_ai = sub[sub[AI_CORE].max(axis=1) == 1]
            has_mand_ai = int((sub_ai['Is_Mandatory'] == 1).any()) if len(sub_ai) else 0
            if n_courses == 0:
                st = 'None'
            elif has['D1'] and ac >= 2 and tc >= 8 and has_mand_ai:
                st = 'Advanced'
            elif ac >= 1:
                st = 'Intermediate'
            else:
                st = 'Foundational-Only'
            rows.append({'University': uni, 'College': coll, 'stage': st,
                         'n_courses': n_courses, 'total_credits': tc,
                         'has_D1': has['D1'], 'aicore_count': ac,
                         'has_mandatory_aicore': has_mand_ai,
                         'Public_Private': m['Public_Private']})
        return pd.DataFrame(rows)

    # Strategy 1: consensus (primary)
    primary = schools
    sens_rows.append({'strategy': 'Consensus (primary)',
                      'n_advanced': int((primary['stage'] == 'Advanced').sum()),
                      'pct_advanced': float((primary['stage'] == 'Advanced').mean() * 100),
                      'n_foundational_only': int((primary['stage'] == 'Foundational-Only').sum()),
                      'n_intermediate': int((primary['stage'] == 'Intermediate').sum()),
                      'n_none': int((primary['stage'] == 'None').sum())})

    # Strategy 2: single-domain priority (D5>D2>D3>D4>D1)
    PRIO = ['D5', 'D2', 'D3', 'D4', 'D1']
    cls_prio = cls.copy()
    for i, d in enumerate(PRIO):
        for d2 in PRIO[i + 1:]:
            cls_prio.loc[cls_prio[d] == 1, d2] = 0
    prio_schools = build_from_alt(cls_prio.rename(columns={}), 'priority')
    sens_rows.append({'strategy': 'Single-domain priority',
                      'n_advanced': int((prio_schools['stage'] == 'Advanced').sum()),
                      'pct_advanced': float((prio_schools['stage'] == 'Advanced').mean() * 100),
                      'n_foundational_only': int((prio_schools['stage'] == 'Foundational-Only').sum()),
                      'n_intermediate': int((prio_schools['stage'] == 'Intermediate').sum()),
                      'n_none': int((prio_schools['stage'] == 'None').sum())})

    # Strategy 3: strict single-domain (drop courses with >=2 domains)
    n_dom = cls[DOMAINS].sum(axis=1)
    cls_strict = cls[n_dom <= 1].copy()
    strict_schools = build_from_alt(cls_strict, 'strict')
    sens_rows.append({'strategy': 'Strict single-domain match',
                      'n_advanced': int((strict_schools['stage'] == 'Advanced').sum()),
                      'pct_advanced': float((strict_schools['stage'] == 'Advanced').mean() * 100),
                      'n_foundational_only': int((strict_schools['stage'] == 'Foundational-Only').sum()),
                      'n_intermediate': int((strict_schools['stage'] == 'Intermediate').sum()),
                      'n_none': int((strict_schools['stage'] == 'None').sum())})

    # Strategy 4: rule-based keyword only
    m1n = m1.rename(columns={
        'D1_Quantitative_Foundations': 'D1', 'D2_AI_ML': 'D2',
        'D3_Data_Science': 'D3', 'D4_Health_Informatics': 'D4',
        'D5_Clinical_AI_Application': 'D5'}).copy()
    m1n['CollegeEN'] = m1n['College'].map(COLLEGE_KO2EN)
    def _remap(row):
        return SCHOOL_NAME_MAP.get((row['University'], row['CollegeEN']),
                                    (row['University'], row['CollegeEN']))
    m1n[['SchoolUni', 'SchoolColl']] = m1n.apply(lambda r: pd.Series(_remap(r)), axis=1)
    rule_schools = build_from_alt(m1n, 'rule')
    sens_rows.append({'strategy': 'Rule-based keyword only',
                      'n_advanced': int((rule_schools['stage'] == 'Advanced').sum()),
                      'pct_advanced': float((rule_schools['stage'] == 'Advanced').mean() * 100),
                      'n_foundational_only': int((rule_schools['stage'] == 'Foundational-Only').sum()),
                      'n_intermediate': int((rule_schools['stage'] == 'Intermediate').sum()),
                      'n_none': int((rule_schools['stage'] == 'None').sum())})

    # Strategy 5: LLM-assisted only
    cls_meta = cls[['Course_ID', 'Credits', 'Is_Mandatory', 'SchoolUni', 'SchoolColl']]
    m2n = m2.merge(cls_meta, on='Course_ID', how='left')
    llm_schools = build_from_alt(m2n, 'llm')
    sens_rows.append({'strategy': 'LLM-assisted only',
                      'n_advanced': int((llm_schools['stage'] == 'Advanced').sum()),
                      'pct_advanced': float((llm_schools['stage'] == 'Advanced').mean() * 100),
                      'n_foundational_only': int((llm_schools['stage'] == 'Foundational-Only').sum()),
                      'n_intermediate': int((llm_schools['stage'] == 'Intermediate').sum()),
                      'n_none': int((llm_schools['stage'] == 'None').sum())})

    pd.DataFrame(sens_rows).to_csv(OUT / 'sensitivity.csv', index=False, encoding='utf-8-sig')

    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # Console summary
    print("\n========== HEADLINE NUMBERS ==========")
    print(f"N schools: {summary['n_schools_total']} ({summary['n_schools_by_college']})")
    print(f"Zero-course schools: {summary['n_zero_course_schools']}")
    print(f"N courses: {summary['n_courses_total']} (unique titles: {summary['n_unique_titles']})")
    print(f"Mean credits (all 63): {summary['mean_credits_all63']:.2f} ± {summary['sd_credits_all63']:.2f}")
    print(f"Mean credits (course-bearing): {summary['mean_credits_courseBearing']:.2f} ± {summary['sd_credits_courseBearing']:.2f}")
    print(f"Quant Foundations credit share: {summary['credit_share_by_domain']['Quantitative Foundations']:.1f}%")
    print(f"Domain school-level adoption (% of 63):")
    for d in DOMAINS:
        print(f"  {DOMAIN_LABEL[d]}: {summary['adoption_pct_of_63'][DOMAIN_LABEL[d]]:.1f}%")
    print(f"Mandatory by domain (% of in-domain courses):")
    for d in DOMAINS:
        m_ = mand_by_domain[DOMAIN_LABEL[d]]
        print(f"  {DOMAIN_LABEL[d]}: {m_['n_mandatory']}/{m_['n_courses_in_domain']} = "
              f"{m_['pct_mandatory']:.1f}%" if m_['pct_mandatory'] is not None else f"  {DOMAIN_LABEL[d]}: n=0")
    print(f"Mandatory chi-square: chi2={summary['mandatory_chi2']['chi2']:.2f}, "
          f"df={summary['mandatory_chi2']['df']}, P={summary['mandatory_chi2']['P']:.3g}")
    print(f"Maturity counts (of 63): {summary['maturity_counts']}")
    print(f"Maturity %: {summary['maturity_pct']}")
    g = summary['advanced_fisher_governance']
    print(f"Governance Advanced Fisher: OR={g['OR']:.2f} (95% CI {g['CI_low']:.2f}-{g['CI_high']:.2f}), P={g['P']:.3g}")
    print(f"  table {g['group_a']} vs {g['group_b']}: {g['table']}")
    r = summary['advanced_fisher_region']
    print(f"Region (Capital vs Non) Advanced Fisher: OR={r['OR']:.2f} (95% CI {r['CI_low']:.2f}-{r['CI_high']:.2f}), P={r['P']:.3g}")
    print(f"  table: {r['table']}")
    print(f"Friedman credits: chi2={summary['friedman_credits']['chi2']:.2f}, P={summary['friedman_credits']['P']:.3g}")
    print(f"\nSensitivity (% Advanced by strategy):")
    for row in sens_rows:
        print(f"  {row['strategy']}: {row['n_advanced']} Advanced ({row['pct_advanced']:.1f}%) | "
              f"None={row['n_none']} | FO={row['n_foundational_only']} | Inter={row['n_intermediate']}")
    print(f"\nGap to next (mean credits + AI-core domains + mandatory AI-core needed):")
    for st, gs in summary['gap_to_next_stage'].items():
        print(f"  {st} -> next: n={gs['n']}, +credits={gs['mean_need_credits']:.2f}±{gs['sd_need_credits']:.2f}, "
              f"+aicore_dom={gs['mean_need_aicore_domains']:.2f}, +mand_ai={gs['mean_need_mandatory_aicore']}")
    print(f"\nThreshold sweep (Option B):")
    for r in sweep:
        print(f"  ≥{r['threshold_credits']} credits: {r['n_advanced']} Advanced ({r['pct_advanced']:.1f}%)")


if __name__ == '__main__':
    run()
