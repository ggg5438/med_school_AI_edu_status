import io, sys, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Paths
PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'results' / 'statistics'
PS_PATH = OUT / 'per_school.csv'
CLS_PATH = PROJECT / 'data' / 'classification' / 'final_adjudicated_classification.csv'

# Constants
DOMAINS = ['D1', 'D2', 'D3', 'D4', 'D5']
DOMAIN_LABEL = {
    'D1': 'Quantitative Foundations', 'D2': 'AI and Machine Learning',
    'D3': 'Data Science', 'D4': 'Health Informatics', 'D5': 'Clinical AI Application',
}
AI_CORE = ['D2', 'D3', 'D5']
COLLEGE_KO2EN = {'의대': 'Medicine', '치대': 'Dentistry', '한의대': 'Korean Medicine'}
SCHOOL_NAME_MAP = {
    ('단국대학교', 'Dentistry'): ('단국대학교 글로컬캠퍼스', 'Dentistry'),
    ('차의과대학교', 'Medicine'): ('차의과학대학교', 'Medicine'),
}


def load_classification():
    df = pd.read_csv(CLS_PATH, encoding='utf-8-sig')
    df['CollegeEN'] = df['College'].map(COLLEGE_KO2EN)
    mand_map = {'필수': 1, '선택': 0}
    df['Is_Mandatory_int'] = df['Is_Mandatory'].map(mand_map).fillna(0).astype(int)
    def remap(row):
        key = (row['University'], row['CollegeEN'])
        return SCHOOL_NAME_MAP.get(key, key)
    df[['SchoolUni', 'SchoolColl']] = df.apply(lambda r: pd.Series(remap(r)), axis=1)
    for d in DOMAINS:
        df[d] = df[d].fillna(0).astype(int)
    df['Credits'] = df['Credits'].fillna(0).astype(float)
    return df


# Advanced vs Foundational-Only extreme-group profiling
def adv_vs_foundational(schools, cls):
    adv = schools[schools['stage'] == 'Advanced'].copy()
    fo = schools[schools['stage'] == 'Foundational-Only'].copy()

    rows = []
    def add(label, adv_vals, fo_vals, test='mwu'):
        adv_a = np.asarray(adv_vals, dtype=float)
        fo_a = np.asarray(fo_vals, dtype=float)
        adv_a = adv_a[~np.isnan(adv_a)]
        fo_a = fo_a[~np.isnan(fo_a)]
        if test == 'mwu' and len(adv_a) and len(fo_a):
            try:
                u, p = stats.mannwhitneyu(adv_a, fo_a, alternative='two-sided')
            except Exception:
                u, p = (np.nan, np.nan)
            r = {
                'metric': label,
                'adv_n': len(adv_a),
                'adv_median': float(np.median(adv_a)) if len(adv_a) else None,
                'adv_mean': float(np.mean(adv_a)) if len(adv_a) else None,
                'adv_sd': float(np.std(adv_a, ddof=1)) if len(adv_a) > 1 else 0.0,
                'fo_n': len(fo_a),
                'fo_median': float(np.median(fo_a)) if len(fo_a) else None,
                'fo_mean': float(np.mean(fo_a)) if len(fo_a) else None,
                'fo_sd': float(np.std(fo_a, ddof=1)) if len(fo_a) > 1 else 0.0,
                'test': 'Mann-Whitney U',
                'statistic': float(u) if np.isfinite(u) else None,
                'P': float(p) if np.isfinite(p) else None,
            }
        elif test == 'prop':
            adv_n = int(adv_a.sum()); adv_d = len(adv_a)
            fo_n = int(fo_a.sum()); fo_d = len(fo_a)
            tab = [[adv_n, adv_d - adv_n], [fo_n, fo_d - fo_n]]
            try:
                _, p = stats.fisher_exact(tab)
            except Exception:
                p = np.nan
            r = {
                'metric': label,
                'adv_n': adv_d, 'adv_mean': float(adv_n / adv_d * 100) if adv_d else None,
                'adv_count': adv_n,
                'fo_n': fo_d, 'fo_mean': float(fo_n / fo_d * 100) if fo_d else None,
                'fo_count': fo_n,
                'test': 'Fisher exact (proportion)',
                'P': float(p) if np.isfinite(p) else None,
            }
        else:
            r = {'metric': label, 'note': 'insufficient data'}
        rows.append(r)

    add('Total credits per school', adv['total_credits'], fo['total_credits'])
    add('AI-core domain count per school', adv['aicore_count'], fo['aicore_count'])
    add('Number of D1 courses', adv['D1_n'], fo['D1_n'])
    add('Number of AI/ML courses', adv['D2_n'], fo['D2_n'])
    add('Number of Data Science courses', adv['D3_n'], fo['D3_n'])
    add('Number of Health Informatics courses', adv['D4_n'], fo['D4_n'])
    add('Number of Clinical AI Application courses', adv['D5_n'], fo['D5_n'])
    add('Has ≥1 mandatory AI-core course (proportion)',
        adv['has_mandatory_aicore'], fo['has_mandatory_aicore'], test='prop')
    schools_set_adv = set(zip(adv['University'], adv['College']))
    schools_set_fo = set(zip(fo['University'], fo['College']))
    cls_adv = cls[cls.apply(lambda r: (r['SchoolUni'], r['SchoolColl']) in schools_set_adv, axis=1)]
    cls_fo = cls[cls.apply(lambda r: (r['SchoolUni'], r['SchoolColl']) in schools_set_fo, axis=1)]
    def per_school_mand(grp_cls, schools_set):
        ratios = []
        for u, c in schools_set:
            sub = grp_cls[(grp_cls['SchoolUni'] == u) & (grp_cls['SchoolColl'] == c)]
            if len(sub):
                ratios.append((sub['Is_Mandatory_int'] == 1).mean())
            else:
                ratios.append(np.nan)
        return np.array(ratios)
    add('Mandatory ratio across all AI/DS courses (per-school median)',
        per_school_mand(cls_adv, schools_set_adv) * 100,
        per_school_mand(cls_fo, schools_set_fo) * 100)
    return pd.DataFrame(rows)


# Ordinal logistic regression on 4-stage maturity
def ordinal_logistic(schools):
    df = schools.copy()
    stage_map = {'None': 0, 'Foundational-Only': 1, 'Intermediate': 2, 'Advanced': 3}
    df['stage_ord'] = df['stage'].map(stage_map)
    df['Is_Public'] = (df['Public_Private'] == '국립').astype(int)
    df['Is_Dentistry'] = (df['College'] == 'Dentistry').astype(int)
    df['Is_KoreanMedicine'] = (df['College'] == 'Korean Medicine').astype(int)
    df['Quota_scaled'] = (df['Admission_Quota'] - df['Admission_Quota'].mean()) / df['Admission_Quota'].std()

    X = df[['Is_Public', 'Capital_Area', 'Is_Dentistry', 'Is_KoreanMedicine', 'Quota_scaled']].astype(float)
    y = df['stage_ord'].astype(int)
    model = OrderedModel(y, X, distr='logit')
    res = model.fit(method='bfgs', disp=0, maxiter=200)
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    ci = res.conf_int(alpha=0.05)
    n = len(y)
    p_k = np.array([(y == k).sum() / n for k in sorted(y.unique())])
    null_ll = float(np.sum([(y == k).sum() * np.log(p) for k, p in zip(sorted(y.unique()), p_k) if p > 0]))
    mcfadden = 1 - res.llf / null_ll

    out_rows = []
    for name in X.columns:
        out_rows.append({
            'variable': name,
            'beta': float(params[name]),
            'OR': float(np.exp(params[name])),
            'SE': float(bse[name]),
            'CI_low': float(np.exp(ci.loc[name, 0])),
            'CI_high': float(np.exp(ci.loc[name, 1])),
            'P': float(pvalues[name]),
        })
    summary = {
        'n_observations': int(len(df)),
        'mcfadden_pseudo_R2': float(mcfadden),
        'log_likelihood': float(res.llf),
        'AIC': float(res.aic),
        'BIC': float(res.bic),
        'levels': ['None', 'Foundational-Only', 'Intermediate', 'Advanced'],
        'reference_profession': 'Medicine',
        'cutpoints': {k: float(params[k]) for k in params.index if k not in X.columns},
    }
    return pd.DataFrame(out_rows), summary


def main():
    schools = pd.read_csv(PS_PATH, encoding='utf-8-sig', keep_default_na=False, na_values=[''])
    schools['stage'] = schools['stage'].astype(str)
    cls = load_classification()
    avf = adv_vs_foundational(schools, cls)
    avf.to_csv(OUT / 'adv_vs_foundational.csv', index=False, encoding='utf-8-sig')
    print("=== Advanced vs Foundational-Only ===")
    print(avf.to_string(index=False))

    ol_df, ol_summary = ordinal_logistic(schools)
    ol_df.to_csv(OUT / 'ordinal_logistic.csv', index=False, encoding='utf-8-sig')
    with open(OUT / 'ordinal_logistic_summary.json', 'w', encoding='utf-8') as f:
        json.dump(ol_summary, f, ensure_ascii=False, indent=2, default=str)
    print("\n=== Ordinal Logistic ===")
    print(f"n={ol_summary['n_observations']}, McFadden pseudo R²={ol_summary['mcfadden_pseudo_R2']:.3f}, "
          f"AIC={ol_summary['AIC']:.2f}")
    print(ol_df.to_string(index=False))


if __name__ == '__main__':
    main()
