# -*- coding: utf-8 -*-
"""
Supplementary analyses (measurement-only).

(a) Extreme-group profiling by total AI/DS credits and by AI-core domain count
    (top vs bottom groups).
(b) Continuous-outcome OLS regression on institutional axes for two outcomes:
    total AI/DS credits per school and AI-core domain count per school
    (predictors: profession dummies + governance + region + scaled quota).
(c) Career-track analyses:
    - track-holding logistic (track present vs absent ~ institutional axes)
    - conditional depth OLS (year-weighted M2-binary | present ~ institutional axes)
    - year-weight function sensitivity (binary vs linear vs clinical-only)

Reads results/statistics/per_school.csv + track_metrics.csv.
Writes outputs to results/statistics/.
"""

import io
import sys
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'statistics'
PS_PATH = OUT / 'per_school.csv'
TRACK_PATH = OUT / 'track_metrics.csv'
CLS_PATH = ROOT / 'data' / 'classification' / 'final_adjudicated_classification.csv'

DOMAINS = ['D1', 'D2', 'D3', 'D4', 'D5']
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


def quartile_extreme_groups(schools, cls, by='total_credits'):
    """Compare top vs bottom groups on descriptive metrics.

    `by` selects the stratifier: 'total_credits' (continuous AI/DS credit sum,
    quartile split) or 'aicore_count' (AI-core domain count: 0 vs >=2).
    """
    s = schools.copy()
    if by == 'total_credits':
        q1, q3 = s['total_credits'].quantile([0.25, 0.75])
        bot = s[s['total_credits'] <= q1].copy()
        top = s[s['total_credits'] >= q3].copy()
        label = 'total credits per school'
    elif by == 'aicore_count':
        bot = s[s['aicore_count'] == 0].copy()
        top = s[s['aicore_count'] >= 2].copy()
        q1, q3 = 0, 2
        label = 'AI-core domain count per school'
    else:
        raise ValueError(by)

    rows = []

    def add(metric, top_vals, bot_vals, test='mwu'):
        ta = np.asarray(top_vals, dtype=float)
        ba = np.asarray(bot_vals, dtype=float)
        ta = ta[~np.isnan(ta)]
        ba = ba[~np.isnan(ba)]
        if test == 'mwu' and len(ta) and len(ba):
            try:
                u, p = stats.mannwhitneyu(ta, ba, alternative='two-sided')
            except Exception:
                u, p = (np.nan, np.nan)
            r = {
                'metric': metric, 'stratifier': by,
                'top_n': len(ta),
                'top_median': float(np.median(ta)) if len(ta) else None,
                'top_mean': float(np.mean(ta)) if len(ta) else None,
                'top_sd': float(np.std(ta, ddof=1)) if len(ta) > 1 else 0.0,
                'bot_n': len(ba),
                'bot_median': float(np.median(ba)) if len(ba) else None,
                'bot_mean': float(np.mean(ba)) if len(ba) else None,
                'bot_sd': float(np.std(ba, ddof=1)) if len(ba) > 1 else 0.0,
                'test': 'Mann-Whitney U',
                'statistic': float(u) if np.isfinite(u) else None,
                'P': float(p) if np.isfinite(p) else None,
            }
        elif test == 'prop':
            t_n = int(ta.sum())
            t_d = len(ta)
            b_n = int(ba.sum())
            b_d = len(ba)
            tab = [[t_n, t_d - t_n], [b_n, b_d - b_n]]
            try:
                _, p = stats.fisher_exact(tab)
            except Exception:
                p = np.nan
            r = {
                'metric': metric, 'stratifier': by,
                'top_n': t_d, 'top_mean': float(t_n / t_d * 100) if t_d else None,
                'top_count': t_n,
                'bot_n': b_d, 'bot_mean': float(b_n / b_d * 100) if b_d else None,
                'bot_count': b_n,
                'test': 'Fisher exact (proportion)',
                'P': float(p) if np.isfinite(p) else None,
            }
        else:
            r = {'metric': metric, 'note': 'insufficient data'}
        rows.append(r)

    add(f'Stratifier: {label}', top[by], bot[by])
    add('Total credits per school', top['total_credits'], bot['total_credits'])
    add('AI-core domain count per school', top['aicore_count'], bot['aicore_count'])
    add('Number of D1 courses', top['D1_n'], bot['D1_n'])
    add('Number of AI/ML courses', top['D2_n'], bot['D2_n'])
    add('Number of Data Science courses', top['D3_n'], bot['D3_n'])
    add('Number of Health Informatics courses', top['D4_n'], bot['D4_n'])
    add('Number of Clinical AI Application courses', top['D5_n'], bot['D5_n'])
    add('Has >=1 mandatory AI-core course (descriptive proportion)',
        top['has_mandatory_aicore'], bot['has_mandatory_aicore'], test='prop')

    schools_set_top = set(zip(top['University'], top['College']))
    schools_set_bot = set(zip(bot['University'], bot['College']))
    cls_top = cls[cls.apply(lambda r: (r['SchoolUni'], r['SchoolColl']) in schools_set_top, axis=1)]
    cls_bot = cls[cls.apply(lambda r: (r['SchoolUni'], r['SchoolColl']) in schools_set_bot, axis=1)]

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
        per_school_mand(cls_top, schools_set_top) * 100,
        per_school_mand(cls_bot, schools_set_bot) * 100)

    out = pd.DataFrame(rows)
    out['top_definition'] = {
        'total_credits': f'>= Q3 ({q3:.1f} credits)',
        'aicore_count': '>=2 AI-core domains',
    }[by]
    out['bot_definition'] = {
        'total_credits': f'<= Q1 ({q1:.1f} credits)',
        'aicore_count': '0 AI-core domains',
    }[by]
    return out


def continuous_outcome_ols(schools, outcome='total_credits'):
    """OLS of a continuous AI/DS curriculum outcome on institutional axes.

    Predictors: governance (public/private), region (Capital area), profession
    dummies (reference: Medicine), scaled admission quota.
    """
    df = schools.copy()
    df['Is_Public'] = (df['Public_Private'] == '국립').astype(int)
    df['Is_Dentistry'] = (df['College'] == 'Dentistry').astype(int)
    df['Is_KoreanMedicine'] = (df['College'] == 'Korean Medicine').astype(int)
    df['Quota_scaled'] = (df['Admission_Quota'] - df['Admission_Quota'].mean()) / df['Admission_Quota'].std()

    pred_cols = ['Is_Public', 'Capital_Area', 'Is_Dentistry', 'Is_KoreanMedicine', 'Quota_scaled']
    X = sm.add_constant(df[pred_cols].astype(float))
    y = df[outcome].astype(float)
    res = sm.OLS(y, X).fit()

    rows = []
    for name in pred_cols:
        ci = res.conf_int().loc[name]
        rows.append({
            'outcome': outcome,
            'variable': name,
            'beta': float(res.params[name]),
            'SE': float(res.bse[name]),
            'CI_low': float(ci[0]),
            'CI_high': float(ci[1]),
            'P': float(res.pvalues[name]),
        })
    summary = {
        'outcome': outcome,
        'n_observations': int(len(df)),
        'R2': float(res.rsquared),
        'adj_R2': float(res.rsquared_adj),
        'F_statistic': float(res.fvalue),
        'F_pvalue': float(res.f_pvalue),
        'AIC': float(res.aic),
        'BIC': float(res.bic),
        'reference_profession': 'Medicine',
        'intercept': float(res.params['const']),
    }
    return pd.DataFrame(rows), summary


def track_depth_by_institution(track_df, schools):
    """For each track (clinical/research): (a) track-holding logistic (present vs
    absent) and (b) conditional M2-binary depth OLS given presence. Predictors:
    profession dummies (ref: Medicine), governance, region, scaled admission quota.
    """
    merged = track_df.merge(
        schools[['University', 'College', 'Public_Private', 'Capital_Area', 'Admission_Quota']],
        on=['University', 'College'], how='left')
    merged['Is_Public'] = (merged['Public_Private'] == '국립').astype(int)
    merged['Is_Dentistry'] = (merged['College'] == 'Dentistry').astype(int)
    merged['Is_KoreanMedicine'] = (merged['College'] == 'Korean Medicine').astype(int)
    quota_mean = merged['Admission_Quota'].mean()
    quota_sd = merged['Admission_Quota'].std()
    merged['Quota_scaled'] = (merged['Admission_Quota'] - quota_mean) / quota_sd

    pred_cols = ['Is_Public', 'Capital_Area', 'Is_Dentistry', 'Is_KoreanMedicine', 'Quota_scaled']

    presence_rows = []
    depth_rows = []
    for tkey in ['clinical', 'research']:
        # (a) Track-holding logistic: track present (M1 crude > 0)
        merged[f'{tkey}_present'] = (merged[f'{tkey}_credits_crude'] > 0).astype(int)
        X = sm.add_constant(merged[pred_cols])
        y_pres = merged[f'{tkey}_present']
        try:
            logit_res = sm.Logit(y_pres, X).fit(disp=0, maxiter=200)
            for name in pred_cols:
                ci = logit_res.conf_int().loc[name]
                presence_rows.append({
                    'track': tkey,
                    'variable': name,
                    'beta': float(logit_res.params[name]),
                    'OR': float(np.exp(logit_res.params[name])),
                    'SE': float(logit_res.bse[name]),
                    'CI_low': float(np.exp(ci[0])),
                    'CI_high': float(np.exp(ci[1])),
                    'P': float(logit_res.pvalues[name]),
                })
            mcf = 1 - logit_res.llf / logit_res.llnull
            presence_rows.append({
                'track': tkey, 'variable': '__model_summary__',
                'n_total': int(len(y_pres)), 'n_present': int(y_pres.sum()),
                'mcfadden_pseudo_R2': float(mcf),
                'AIC': float(logit_res.aic),
            })
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as e:
            presence_rows.append({
                'track': tkey, 'variable': '__model_error__', 'P': None,
                'note': type(e).__name__,
            })

        # (b) Conditional depth OLS: among schools with the track present,
        #     regress M2-binary on the same predictors.
        sub = merged[merged[f'{tkey}_present'] == 1].copy()
        Xc = sm.add_constant(sub[pred_cols])
        yc = sub[f'{tkey}_M2_binary']
        if len(sub) > len(pred_cols) + 2:
            try:
                ols_res = sm.OLS(yc, Xc).fit()
                for name in pred_cols:
                    ci = ols_res.conf_int().loc[name]
                    depth_rows.append({
                        'track': tkey,
                        'variable': name,
                        'beta': float(ols_res.params[name]),
                        'SE': float(ols_res.bse[name]),
                        'CI_low': float(ci[0]),
                        'CI_high': float(ci[1]),
                        'P': float(ols_res.pvalues[name]),
                    })
                depth_rows.append({
                    'track': tkey, 'variable': '__model_summary__',
                    'n_obs': int(len(sub)),
                    'R2': float(ols_res.rsquared),
                    'adj_R2': float(ols_res.rsquared_adj),
                    'F_pvalue': float(ols_res.f_pvalue),
                })
            except Exception as e:
                depth_rows.append({
                    'track': tkey, 'variable': '__model_error__', 'P': None,
                    'note': str(type(e).__name__),
                })
        else:
            depth_rows.append({
                'track': tkey, 'variable': '__model_skipped__',
                'note': f'n={len(sub)} too small for OLS with {len(pred_cols)} predictors',
            })

    return (pd.DataFrame(presence_rows), pd.DataFrame(depth_rows))


def track_weight_sensitivity(track_df):
    """Compare three year-weight functions for the two career tracks (school-level)."""
    rows = []
    for tkey in ['clinical', 'research']:
        for wname in ['binary', 'linear', 'clin_only']:
            col = f'{tkey}_M2_{wname}'
            v = track_df[col].astype(float)
            rows.append({
                'track': tkey,
                'weight_function': wname,
                'n_schools': int(len(v)),
                'mean': float(v.mean()),
                'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                'median': float(v.median()),
                'q25': float(v.quantile(0.25)),
                'q75': float(v.quantile(0.75)),
                'n_zero': int((v == 0).sum()),
            })
    pairwise = []
    for wname in ['binary', 'linear', 'clin_only']:
        cli = track_df[f'clinical_M2_{wname}'].values
        res = track_df[f'research_M2_{wname}'].values
        try:
            w = stats.wilcoxon(cli, res, zero_method='wilcox', alternative='two-sided', method='approx')
            pairwise.append({'weight_function': wname,
                             'stat': float(w.statistic), 'P': float(w.pvalue)})
        except Exception:
            pairwise.append({'weight_function': wname, 'stat': None, 'P': None})
    rank_corr = []
    for tkey in ['clinical', 'research']:
        for w1, w2 in [('binary', 'linear'), ('binary', 'clin_only'), ('linear', 'clin_only')]:
            v1 = track_df[f'{tkey}_M2_{w1}']
            v2 = track_df[f'{tkey}_M2_{w2}']
            rho, p = stats.spearmanr(v1, v2)
            rank_corr.append({'track': tkey, 'pair': f'{w1}_vs_{w2}',
                              'spearman_rho': float(rho), 'P': float(p)})
    return pd.DataFrame(rows), pd.DataFrame(pairwise), pd.DataFrame(rank_corr)


def main():
    schools = pd.read_csv(PS_PATH, encoding='utf-8-sig', keep_default_na=False, na_values=[''])
    cls = load_classification()
    track_df = pd.read_csv(TRACK_PATH, encoding='utf-8-sig')

    print("=== Extreme-group profiling (top vs bottom) ===")
    for by in ['total_credits', 'aicore_count']:
        q = quartile_extreme_groups(schools, cls, by=by)
        q.to_csv(OUT / f'quartile_extremes_by_{by}.csv', index=False, encoding='utf-8-sig')
        print(f"\n--- Stratified by {by} ---")
        print(q.to_string(index=False))

    print("\n=== Continuous-outcome OLS ===")
    for outcome in ['total_credits', 'aicore_count']:
        ols_df, ols_sum = continuous_outcome_ols(schools, outcome=outcome)
        ols_df.to_csv(OUT / f'continuous_ols_{outcome}.csv', index=False, encoding='utf-8-sig')
        with open(OUT / f'continuous_ols_{outcome}_summary.json', 'w', encoding='utf-8') as f:
            json.dump(ols_sum, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n--- Outcome: {outcome} ---")
        print(f"n={ols_sum['n_observations']}, R2={ols_sum['R2']:.3f}, "
              f"adj R2={ols_sum['adj_R2']:.3f}, F P={ols_sum['F_pvalue']:.3g}, "
              f"AIC={ols_sum['AIC']:.2f}")
        print(ols_df.to_string(index=False))

    presence_df, depth_df = track_depth_by_institution(track_df, schools)
    presence_df.to_csv(OUT / 'track_presence_logistic.csv', index=False, encoding='utf-8-sig')
    depth_df.to_csv(OUT / 'track_depth_ols.csv', index=False, encoding='utf-8-sig')
    print("\n=== Track holding by institutional axes (logistic) ===")
    print(presence_df.to_string(index=False))
    print("\n=== Track conditional depth (OLS on M2-binary | track present) ===")
    print(depth_df.to_string(index=False))

    sens_df, pair_df, rank_df = track_weight_sensitivity(track_df)
    sens_df.to_csv(OUT / 'track_weight_sensitivity.csv', index=False, encoding='utf-8-sig')
    pair_df.to_csv(OUT / 'track_wilcoxon_by_weight.csv', index=False, encoding='utf-8-sig')
    rank_df.to_csv(OUT / 'track_rank_correlation.csv', index=False, encoding='utf-8-sig')
    print("\n=== Track weight-function sensitivity ===")
    print(sens_df.to_string(index=False))
    print("\n--- Wilcoxon clinical vs research under each weight function ---")
    print(pair_df.to_string(index=False))
    print("\n--- Spearman rank correlation between weight functions (same track) ---")
    print(rank_df.to_string(index=False))


if __name__ == '__main__':
    main()
