# Core statistical analyses for the AI/DS curriculum study.

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

import config

warnings.filterwarnings('ignore', category=FutureWarning)


def _holm_adjust(p_values):
    _, p_adj, _, _ = multipletests(p_values, method='holm')
    return p_adj


# A. Descriptive Statistics (RQ1)

def descriptive_stats(course_df, uni_df):
    results = {}

    results['n_courses'] = len(course_df)
    results['n_universities'] = len(uni_df)
    results['total_credits'] = course_df['Credits'].sum()
    results['mean_credits_per_course'] = course_df['Credits'].mean()
    results['mandatory_ratio'] = course_df['Is_Mandatory_Binary'].mean()

    college_stats = []
    for college in config.COLLEGE_ORDER:
        cmask = uni_df['College'] == college
        cuni = uni_df[cmask]
        n = len(cuni)
        ccourses = course_df[course_df['College'] == college]

        row = {
            'College': college,
            'N_Schools': n,
            'N_Courses': len(ccourses),
            'Total_Credits_Mean': cuni['Total_Credits'].mean(),
            'Total_Credits_SD': cuni['Total_Credits'].std(),
            'Courses_per_School_Mean': cuni['Total_Courses'].mean(),
            'Courses_per_School_SD': cuni['Total_Courses'].std(),
        }
        for d in config.DOMAIN_COLS:
            row[f'{d}_Pct'] = cuni[f'Has_{d}'].mean() * 100
        college_stats.append(row)

    total = {
        'College': 'Total',
        'N_Schools': len(uni_df),
        'N_Courses': len(course_df),
        'Total_Credits_Mean': uni_df['Total_Credits'].mean(),
        'Total_Credits_SD': uni_df['Total_Credits'].std(),
        'Courses_per_School_Mean': uni_df['Total_Courses'].mean(),
        'Courses_per_School_SD': uni_df['Total_Courses'].std(),
    }
    for d in config.DOMAIN_COLS:
        total[f'{d}_Pct'] = uni_df[f'Has_{d}'].mean() * 100
    college_stats.append(total)

    results['table1'] = pd.DataFrame(college_stats)

    domain_dist = []
    for d in config.DOMAIN_COLS:
        cr = uni_df[f'{d}_Credits'].sum()
        domain_dist.append({
            'Domain': config.DOMAIN_LABELS[d],
            'Total_Credits': cr,
            'Pct_Credits': cr / results['total_credits'] * 100,
            'N_Schools': uni_df[f'Has_{d}'].sum(),
            'Pct_Schools': uni_df[f'Has_{d}'].mean() * 100,
        })
    results['domain_distribution'] = pd.DataFrame(domain_dist)

    return results


# B. Group Comparisons (RQ2)

def college_anova(uni_df):
    results = {}
    groups = [uni_df[uni_df['College'] == c] for c in config.COLLEGE_ORDER]

    outcomes = ['Total_Credits', 'Total_Courses', 'Credits_per_Course']
    anova_rows = []
    for var in outcomes:
        vals = [g[var].values for g in groups]
        F, p = stats.f_oneway(*vals)
        grand_mean = uni_df[var].mean()
        ss_between = sum(len(g) * (g[var].mean() - grand_mean)**2 for g in groups)
        ss_total = sum((uni_df[var] - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        anova_rows.append({
            'Variable': var, 'F': F, 'p': p, 'eta_squared': eta_sq,
            **{f'{c}_mean': g[var].mean() for c, g in zip(config.COLLEGE_ORDER, groups)},
        })
    anova_main = pd.DataFrame(anova_rows)
    anova_main['p_adjusted'] = _holm_adjust(anova_main['p'].values)
    results['anova_main'] = anova_main

    domain_rows = []
    for d in config.DOMAIN_COLS:
        var = f'{d}_Credits'
        vals = [g[var].values for g in groups]
        F, p = stats.f_oneway(*vals)
        grand_mean = uni_df[var].mean()
        ss_between = sum(len(g) * (g[var].mean() - grand_mean)**2 for g in groups)
        ss_total = sum((uni_df[var] - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        domain_rows.append({
            'Domain': config.DOMAIN_LABELS[d], 'F': F, 'p': p, 'eta_squared': eta_sq,
        })
    anova_domains = pd.DataFrame(domain_rows)
    anova_domains['p_adjusted'] = _holm_adjust(anova_domains['p'].values)
    results['anova_domains'] = anova_domains

    return results


def public_private_test(uni_df):
    pub = uni_df[uni_df['Is_Public'] == 1]
    priv = uni_df[uni_df['Is_Public'] == 0]
    results = {}
    rows = []

    for var in ['Total_Credits', 'Total_Courses', 'Shannon_Diversity', 'Mandatory_Ratio']:
        t, p = stats.ttest_ind(pub[var], priv[var])
        pooled_std = np.sqrt(((len(pub)-1)*pub[var].std()**2 + (len(priv)-1)*priv[var].std()**2)
                             / (len(pub)+len(priv)-2))
        d = (pub[var].mean() - priv[var].mean()) / pooled_std if pooled_std > 0 else 0
        rows.append({
            'Variable': var,
            'Public_Mean': pub[var].mean(), 'Public_SD': pub[var].std(),
            'Private_Mean': priv[var].mean(), 'Private_SD': priv[var].std(),
            't': t, 'p': p, 'Cohens_d': d,
        })
    pp_df = pd.DataFrame(rows)
    pp_df['p_adjusted'] = _holm_adjust(pp_df['p'].values)
    results['pp_tests'] = pp_df
    return results


# C. Temporal Placement Patterns

def temporal_placement(course_df):
    results = []
    for d in config.DOMAIN_COLS:
        ddata = course_df[course_df[d] == 1]
        n_total = len(ddata)
        if n_total == 0:
            continue
        n_pre = int(ddata['Pre_Medical'].sum())
        n_med = int(ddata['Medical'].sum())
        n_both = int(((ddata['Pre_Medical'] == 1) & (ddata['Medical'] == 1)).sum())
        results.append({
            'Domain': config.DOMAIN_LABELS[d],
            'N_Courses': n_total,
            'N_Pre_Medical': n_pre,
            'N_Medical': n_med,
            'N_Both': n_both,
            'Pct_Pre_Medical': n_pre / n_total * 100,
            'Pct_Medical': n_med / n_total * 100,
        })

    temporal_df = pd.DataFrame(results)

    cont_rows = []
    for d in config.DOMAIN_COLS:
        ddata = course_df[course_df[d] == 1]
        n_pre = int(ddata['Pre_Medical'].sum())
        n_med = int(ddata['Medical'].sum())
        cont_rows.append([n_pre, n_med])

    cont_table = pd.DataFrame(
        cont_rows,
        index=[config.DOMAIN_SHORT[d] for d in config.DOMAIN_COLS],
        columns=['Pre-Medical', 'Medical'],
    )
    if cont_table.sum().min() > 0 and cont_table.sum(axis=1).min() > 0:
        chi2, p, dof, _ = stats.chi2_contingency(cont_table)
        n_obs = cont_table.values.sum()
        k = min(cont_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n_obs * k)) if n_obs > 0 and k > 0 else 0
    else:
        chi2, p, dof, cramers_v = np.nan, np.nan, np.nan, np.nan

    return {
        'temporal_df': temporal_df,
        'contingency': cont_table,
        'chi2': chi2, 'chi2_p': p, 'chi2_dof': dof, 'cramers_v': cramers_v,
    }


# D. Mandatory/Elective Patterns

def mandatory_patterns(course_df, uni_df):
    domain_mand = []
    for d in config.DOMAIN_COLS:
        ddata = course_df[course_df[d] == 1]
        n_total = len(ddata)
        if n_total == 0:
            continue
        n_mand = ddata['Is_Mandatory_Binary'].sum()
        cr_mand = ddata[ddata['Is_Mandatory_Binary'] == 1]['Credits'].sum()
        cr_total = ddata['Credits'].sum()
        domain_mand.append({
            'Domain': config.DOMAIN_LABELS[d],
            'N_Courses': n_total,
            'N_Mandatory': int(n_mand),
            'Pct_Mandatory': n_mand / n_total * 100,
            'Credits_Mandatory': cr_mand,
            'Credits_Total': cr_total,
            'Pct_Credits_Mandatory': cr_mand / cr_total * 100 if cr_total > 0 else 0,
        })
    domain_mand_df = pd.DataFrame(domain_mand)

    college_mand = []
    for college in config.COLLEGE_ORDER:
        cdata = course_df[course_df['College'] == college]
        n_total = len(cdata)
        n_mand = cdata['Is_Mandatory_Binary'].sum()
        college_mand.append({
            'College': college,
            'N_Courses': n_total,
            'N_Mandatory': int(n_mand),
            'Pct_Mandatory': n_mand / n_total * 100 if n_total > 0 else 0,
        })
    college_mand_df = pd.DataFrame(college_mand)

    r, p = stats.pearsonr(uni_df['Mandatory_Ratio'], uni_df['Total_Credits'])

    cont_rows = []
    for d in config.DOMAIN_COLS:
        ddata = course_df[course_df[d] == 1]
        n_mand = int(ddata['Is_Mandatory_Binary'].sum())
        n_elec = len(ddata) - n_mand
        cont_rows.append([n_mand, n_elec])
    mandatory_contingency = pd.DataFrame(
        cont_rows,
        index=[config.DOMAIN_SHORT[d] for d in config.DOMAIN_COLS],
        columns=['Mandatory', 'Elective'],
    )
    chi2_mand, p_mand, dof_mand, _ = stats.chi2_contingency(mandatory_contingency)
    n_obs = mandatory_contingency.values.sum()
    k = min(mandatory_contingency.shape) - 1
    cramers_v_mand = np.sqrt(chi2_mand / (n_obs * k)) if n_obs > 0 and k > 0 else 0

    return {
        'domain_mandatory': domain_mand_df,
        'college_mandatory': college_mand_df,
        'mandatory_credits_corr': {'r': r, 'p': p},
        'mandatory_chi2': chi2_mand,
        'mandatory_chi2_p': p_mand,
        'mandatory_chi2_dof': dof_mand,
        'mandatory_cramers_v': cramers_v_mand,
        'mandatory_contingency': mandatory_contingency,
    }


# E. Curriculum Maturity Analysis

def maturity_analysis(uni_df):
    dist = uni_df['Maturity_Label'].value_counts()

    cross = pd.crosstab(uni_df['College'], uni_df['Maturity_Label'])

    pp_cross = pd.crosstab(uni_df['Is_Public'].map({1: 'Public', 0: 'Private'}),
                           uni_df['Maturity_Label'])

    maturity_stats = uni_df.groupby('Maturity_Label').agg({
        'Total_Credits': ['mean', 'std', 'count'],
        'Breadth': 'mean',
        'AI_Core_Ratio': 'mean',
    }).round(2)

    groups_college = [uni_df[uni_df['College'] == c]['Maturity_Level']
                      for c in config.COLLEGE_ORDER]
    H_college, p_college = stats.kruskal(*groups_college)

    pub = uni_df[uni_df['Is_Public'] == 1]['Maturity_Level']
    priv = uni_df[uni_df['Is_Public'] == 0]['Maturity_Level']
    U, p_pp = stats.mannwhitneyu(pub, priv, alternative='two-sided')

    return {
        'distribution': dist,
        'college_cross': cross,
        'pp_cross': pp_cross,
        'maturity_stats': maturity_stats,
        'kruskal_college': {'H': H_college, 'p': p_college},
        'mannwhitney_pp': {'U': U, 'p': p_pp},
    }


# F. Maturity Predictors (ordinal logistic + Fisher exact)

def maturity_predictors(uni_df):
    results = {}

    # Ordinal logistic regression
    df = uni_df.dropna(subset=['Admission_Quota']).copy()
    df['Maturity_Level'] = df['Maturity_Level'].astype(float)

    college_dum = pd.get_dummies(df['College'], drop_first=True, dtype=float)
    X = pd.concat([college_dum, df[['Is_Public', 'Admission_Quota']].astype(float)], axis=1)
    y = df['Maturity_Level']

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        mod = OrderedModel(y, X, distr='logit')
        res = mod.fit(method='bfgs', disp=False)

        coef_df = pd.DataFrame({
            'Variable': res.params.index,
            'Coefficient': res.params.values,
            'SE': res.bse.values,
            'p_value': res.pvalues.values,
        })
        results['ordinal_logistic'] = coef_df
        results['ordinal_summary'] = {
            'pseudo_r_squared': float(res.prsquared),
            'log_likelihood': float(res.llf),
            'n': int(len(df)),
            'converged': bool(res.mle_retvals.get('converged', False)
                             if hasattr(res, 'mle_retvals') else True),
            'note': 'Exploratory analysis; reported in Multimedia Appendix only due to low pseudo R-squared',
        }
    except Exception as e:
        results['ordinal_logistic'] = pd.DataFrame()
        results['ordinal_summary'] = {'error': str(e)}

    # Fisher exact test: Advanced vs non-Advanced by governance
    df_full = uni_df.copy()
    df_full['Is_Advanced'] = (df_full['Maturity_Level'] >= 3).astype(int)
    ct = pd.crosstab(df_full['Is_Public'].map({1: 'Public', 0: 'Private'}),
                     df_full['Is_Advanced'].map({1: 'Advanced', 0: 'Other'}))
    ct = ct.reindex(index=['Public', 'Private'], columns=['Advanced', 'Other'])

    if ct.shape == (2, 2) and not ct.isna().any().any():
        ct_arr = ct.values
        odds_ratio, fisher_p = stats.fisher_exact(ct_arr, alternative='two-sided')
        try:
            from scipy.stats.contingency import odds_ratio as _sp_odds_ratio
            _res = _sp_odds_ratio(ct_arr, kind='conditional')
            _ci = _res.confidence_interval(confidence_level=0.95)
            ci_low = float(_ci.low)
            ci_high = float(_ci.high)
            or_cond = float(_res.statistic)
            ci_method = ("exact conditional MLE "
                         "(scipy.stats.contingency.odds_ratio, kind='conditional')")
        except Exception:
            ci_low, ci_high, or_cond, ci_method = (np.nan, np.nan, np.nan,
                                                   'unavailable (scipy<1.11)')
    else:
        odds_ratio, fisher_p = np.nan, np.nan
        ci_low, ci_high, or_cond, ci_method = np.nan, np.nan, np.nan, 'n/a'

    results['fisher_exact'] = {
        'contingency': ct,
        'odds_ratio': float(odds_ratio) if not np.isnan(odds_ratio) else None,
        'odds_ratio_sample': float(odds_ratio) if not np.isnan(odds_ratio) else None,
        'odds_ratio_conditional_mle': (float(or_cond)
                                        if not np.isnan(or_cond) else None),
        'odds_ratio_reference': 'private',
        'odds_ratio_ci_low': float(ci_low) if not np.isnan(ci_low) else None,
        'odds_ratio_ci_high': float(ci_high) if not np.isnan(ci_high) else None,
        'ci_method': ci_method,
        'p_value': float(fisher_p) if not np.isnan(fisher_p) else None,
    }

    return results


# G. Maturity Feature Comparison (Advanced vs Foundational-Only)

def maturity_feature_comparison(uni_df):
    advanced = uni_df[uni_df['Maturity_Level'] == 3]
    found_only = uni_df[uni_df['Maturity_Level'] == 1]

    compare_vars = [
        'Total_Credits', 'Total_Courses', 'Credits_per_Course',
        'Breadth', 'Mandatory_Ratio', 'Pre_Medical_Ratio',
        'Admission_Quota', 'Is_Public',
    ]

    rows = []
    for var in compare_vars:
        adv_vals = advanced[var].dropna()
        fo_vals = found_only[var].dropna()
        if len(adv_vals) < 2 or len(fo_vals) < 2:
            rows.append({'Variable': var, 'n_adv': len(adv_vals), 'n_fo': len(fo_vals)})
            continue

        U, p = stats.mannwhitneyu(adv_vals, fo_vals, alternative='two-sided')
        n1, n2 = len(adv_vals), len(fo_vals)
        r_rb = 1 - (2 * U) / (n1 * n2)

        rows.append({
            'Variable': var,
            'Advanced_Median': adv_vals.median(),
            'Advanced_IQR': f"{adv_vals.quantile(0.25):.1f}-{adv_vals.quantile(0.75):.1f}",
            'FoundOnly_Median': fo_vals.median(),
            'FoundOnly_IQR': f"{fo_vals.quantile(0.25):.1f}-{fo_vals.quantile(0.75):.1f}",
            'U': U, 'p_value': p,
            'rank_biserial_r': r_rb,
            'n_adv': n1, 'n_fo': n2,
        })

    return pd.DataFrame(rows)


# H. Gap Analysis (distance to next maturity stage)

def gap_analysis(uni_df):
    rows = []
    for _, row in uni_df.iterrows():
        level = row['Maturity_Level']
        n_ai_domains = sum(row[f'Has_{d}'] for d in config.AI_CORE_DOMAINS)
        total_cr = row['Total_Credits']
        has_found = row['Has_Foundational']
        has_ai = row['Has_AI_Core']

        if level == 3:
            needed_domains = 0
            needed_credits = 0
            next_level = 'Already Advanced'
        elif level == 2:
            needed_domains = max(0, 2 - n_ai_domains)
            needed_credits = max(0, 8 - total_cr)
            next_level = 'Advanced'
        elif level == 1:
            needed_domains = 1
            needed_credits = max(0, 2)
            next_level = 'Intermediate'
        else:
            needed_domains = 1
            needed_credits = 2
            next_level = 'Foundational-Only'

        rows.append({
            'University': row['University'],
            'College': row['College'],
            'Current_Level': row['Maturity_Label'],
            'Current_Credits': total_cr,
            'Current_AI_Domains': n_ai_domains,
            'Next_Level': next_level,
            'Additional_Domains_Needed': needed_domains,
            'Additional_Credits_Needed': needed_credits,
        })

    gap_df = pd.DataFrame(rows)

    summary = gap_df.groupby('Current_Level').agg({
        'Additional_Domains_Needed': ['median', 'mean'],
        'Additional_Credits_Needed': ['median', 'mean'],
        'University': 'count',
    }).round(1)
    summary.columns = ['Domains_Median', 'Domains_Mean',
                        'Credits_Median', 'Credits_Mean', 'N_Schools']

    return {'gap_df': gap_df, 'summary': summary}


# I. Effect Size Summary

def effect_size_summary(results):
    rows = []

    anova = results['anova']['anova_main']
    for _, r in anova.iterrows():
        rows.append({
            'Test': f'One-way ANOVA: {r["Variable"]}',
            'Statistic': f'F={r["F"]:.2f}',
            'p_value': r['p'],
            'p_adjusted': r['p_adjusted'],
            'Effect_Size': f'eta²={r["eta_squared"]:.3f}',
            'Interpretation': 'Large' if r['eta_squared'] > 0.14 else
                             'Medium' if r['eta_squared'] > 0.06 else 'Small',
        })

    pp = results['public_private']['pp_tests']
    for _, r in pp.iterrows():
        rows.append({
            'Test': f't-test: {r["Variable"]} (Pub vs Priv)',
            'Statistic': f't={r["t"]:.2f}',
            'p_value': r['p'],
            'p_adjusted': r['p_adjusted'],
            'Effect_Size': f'd={r["Cohens_d"]:.2f}',
            'Interpretation': 'Large' if abs(r['Cohens_d']) > 0.8 else
                             'Medium' if abs(r['Cohens_d']) > 0.5 else 'Small',
        })

    mand = results['mandatory']
    rows.append({
        'Test': 'Chi-square: Domain × Mandatory',
        'Statistic': f'chi²={mand["mandatory_chi2"]:.1f}',
        'p_value': mand['mandatory_chi2_p'],
        'p_adjusted': mand['mandatory_chi2_p'],
        'Effect_Size': f'V={mand["mandatory_cramers_v"]:.3f}',
        'Interpretation': 'Large' if mand['mandatory_cramers_v'] > 0.35 else
                         'Medium' if mand['mandatory_cramers_v'] > 0.15 else 'Small',
    })

    temp = results['temporal']
    rows.append({
        'Test': 'Chi-square: Domain × Phase',
        'Statistic': f'chi²={temp["chi2"]:.2f}',
        'p_value': temp['chi2_p'],
        'p_adjusted': temp['chi2_p'],
        'Effect_Size': f'V={temp["cramers_v"]:.3f}',
        'Interpretation': 'Large' if temp['cramers_v'] > 0.35 else
                         'Medium' if temp['cramers_v'] > 0.15 else 'Small',
    })

    mat = results['maturity']
    rows.append({
        'Test': 'Kruskal-Wallis: Maturity by College',
        'Statistic': f'H={mat["kruskal_college"]["H"]:.2f}',
        'p_value': mat['kruskal_college']['p'],
        'p_adjusted': mat['kruskal_college']['p'],
        'Effect_Size': '-',
        'Interpretation': 'Significant' if mat['kruskal_college']['p'] < 0.05 else 'Non-significant',
    })

    rows.append({
        'Test': 'Mann-Whitney U: Maturity by Governance',
        'Statistic': f'U={mat["mannwhitney_pp"]["U"]:.0f}',
        'p_value': mat['mannwhitney_pp']['p'],
        'p_adjusted': mat['mannwhitney_pp']['p'],
        'Effect_Size': '-',
        'Interpretation': 'Significant' if mat['mannwhitney_pp']['p'] < 0.05 else 'Non-significant',
    })

    if 'maturity_predictors' in results:
        mp = results['maturity_predictors']
        fe = mp.get('fisher_exact', {})
        if fe.get('p_value') is not None:
            _or = fe['odds_ratio']
            _lo = fe.get('odds_ratio_ci_low')
            _hi = fe.get('odds_ratio_ci_high')
            if _lo is not None and _hi is not None:
                _effect = f'OR={_or:.2f} (95% CI {_lo:.2f}-{_hi:.2f})'
            else:
                _effect = f'OR={_or:.2f}'
            rows.append({
                'Test': 'Fisher exact: Advanced × Governance (ref: private)',
                'Statistic': _effect,
                'p_value': fe['p_value'],
                'p_adjusted': fe['p_value'],
                'Effect_Size': _effect,
                'Interpretation': 'Significant' if fe['p_value'] < 0.05 else 'Non-significant',
            })

    if 'maturity_comparison' in results:
        mc = results['maturity_comparison']
        mc_with_p = mc.dropna(subset=['p_value'])
        if len(mc_with_p) > 1:
            mc_adj = _holm_adjust(mc_with_p['p_value'].values)
            for idx, (_, r) in enumerate(mc_with_p.iterrows()):
                rows.append({
                    'Test': f'Mann-Whitney U: {r["Variable"]} (Adv vs FO)',
                    'Statistic': f'U={r["U"]:.1f}',
                    'p_value': r['p_value'],
                    'p_adjusted': mc_adj[idx],
                    'Effect_Size': f'r={r["rank_biserial_r"]:.2f}',
                    'Interpretation': 'Large' if abs(r['rank_biserial_r']) > 0.5 else
                                     'Medium' if abs(r['rank_biserial_r']) > 0.3 else 'Small',
                })

    return pd.DataFrame(rows)


# Master runner

def run_all(course_df, uni_df):
    print("  A. Descriptive statistics...")
    desc = descriptive_stats(course_df, uni_df)

    print("  B. College ANOVA...")
    anova = college_anova(uni_df)

    print("  B. Public/Private tests...")
    pp = public_private_test(uni_df)

    print("  C. Temporal placement...")
    temp = temporal_placement(course_df)

    print("  D. Mandatory patterns...")
    mand = mandatory_patterns(course_df, uni_df)

    print("  E. Maturity analysis...")
    mat = maturity_analysis(uni_df)

    print("  F. Maturity predictors...")
    mat_pred = maturity_predictors(uni_df)

    print("  G. Maturity feature comparison...")
    mat_comp = maturity_feature_comparison(uni_df)

    print("  H. Gap analysis...")
    gap = gap_analysis(uni_df)

    results = {
        'descriptive': desc,
        'anova': anova,
        'public_private': pp,
        'temporal': temp,
        'mandatory': mand,
        'maturity': mat,
        'maturity_predictors': mat_pred,
        'maturity_comparison': mat_comp,
        'gap_analysis': gap,
    }

    print("  I. Effect size summary...")
    results['effect_sizes'] = effect_size_summary(results)

    return results
