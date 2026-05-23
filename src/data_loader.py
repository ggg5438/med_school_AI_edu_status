# Data loading, preprocessing, domain classification, and university feature matrix.

import numpy as np
import pandas as pd
import config


# Domain classification

def classify_course(course_name: str) -> dict:
    name = str(course_name).strip()

    name = config.TYPO_CORRECTIONS.get(name, name)

    result = {}
    for domain, keywords in config.DOMAIN_KEYWORDS.items():
        matched = any(kw.lower() in name.lower() for kw in keywords)
        result[domain] = 1 if matched else 0

    original = str(course_name).strip()
    if original in config.MANUAL_OVERRIDES:
        for d, v in config.MANUAL_OVERRIDES[original].items():
            result[d] = v

    return result


# Data loading

def load_curriculum() -> pd.DataFrame:
    df = pd.read_excel(config.CURRICULUM_FILE)

    rename = {k: v for k, v in config.CURRICULUM_COL_RENAME.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)

    df['Course_Name'] = df['Course_Name'].replace(config.TYPO_CORRECTIONS)

    df = df[~df['Course_Name'].isin(config.EXCLUDE_COURSES)].copy()

    df['College'] = df['College'].map(config.COLLEGE_MAP)

    df['University'] = df['University'].replace(config.UNIVERSITY_NAME_MAP)
    for (col_en, old_name), new_name in config.UNIVERSITY_NAME_MAP_BY_COLLEGE.items():
        mask = (df['College'] == col_en) & (df['University'] == old_name)
        df.loc[mask, 'University'] = new_name

    domain_flags = df['Course_Name'].apply(classify_course).apply(pd.Series)
    df = pd.concat([df, domain_flags], axis=1)

    df['Year_Stage'] = df['Year_Raw'].apply(_parse_year_stage)

    df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce').fillna(0)

    df['Is_Mandatory_Binary'] = (df['Is_Mandatory'] == '필수').astype(int)

    return df


def _parse_year_stage(raw):
    if pd.isna(raw):
        return None
    s = str(raw).strip()

    vague_patterns = ['예과', '본과', '전학년', '대학원']
    for pat in vague_patterns:
        if s.startswith(pat) and len(s.replace(' ', '')) <= len(pat) + 3:
            return None
    if '~' in s:
        return None
    if '본과1-4' in s.replace(' ', ''):
        return None
    if ',' in s:
        return None

    mapping = {
        '예 1-1': 'Pre-1', '예 1-2': 'Pre-1',
        '예 1': 'Pre-1', '예1': 'Pre-1', '예1-1': 'Pre-1', '예1-2': 'Pre-1',
        '예 2-1': 'Pre-2', '예 2-2': 'Pre-2',
        '예 2': 'Pre-2', '예2': 'Pre-2', '예2-1': 'Pre-2', '예2-2': 'Pre-2',
        '본 1-1': 'Med-1', '본 1-2': 'Med-1', '본과 1-2': 'Med-1',
        '본1': 'Med-1', '본1-1': 'Med-1', '본1-2': 'Med-1',
        '본 2-1': 'Med-2', '본 2-2': 'Med-2',
        '본2': 'Med-2', '본2-1': 'Med-2', '본2-2': 'Med-2',
        '본 3-1': 'Med-3', '본 3-2': 'Med-3', '본 3': 'Med-3', '본3': 'Med-3',
        '본 4-1': 'Med-4', '본 4-2': 'Med-4', '본 4': 'Med-4', '본4': 'Med-4',
        '예과 1학기': 'Pre-1', '예과 2학기': 'Pre-2',
    }
    s_clean = s.strip()
    if s_clean in mapping:
        return mapping[s_clean]

    if '학사과정' in s_clean:
        for digit in ['2', '3', '4']:
            if digit in s_clean:
                return f'Med-{int(digit)-1}' if int(digit) <= 4 else None
        return None

    s_lower = s_clean.replace(' ', '')
    if s_lower.startswith('예'):
        for digit in ['2', '1']:
            if digit in s_lower[1:]:
                return f'Pre-{digit}'
    if s_lower.startswith('본'):
        for digit in ['4', '3', '2', '1']:
            if digit in s_lower[1:]:
                return f'Med-{digit}'

    return None


def _parse_year_stage_list(raw):
    if pd.isna(raw):
        return []
    s = str(raw).strip()

    single = _parse_year_stage(raw)
    if single is not None:
        return [single]

    stages = []
    s_clean = s.replace(' ', '')
    if '예~본' in s_clean or s_clean == '전학년':
        stages = ['Pre-1', 'Pre-2', 'Med-1', 'Med-2', 'Med-3', 'Med-4']
    elif s_clean.startswith('본') and '~' in s_clean:
        stages = ['Med-1', 'Med-2', 'Med-3', 'Med-4']
    elif s_clean.startswith('예과'):
        stages = ['Pre-1', 'Pre-2']
    elif s_clean.startswith('본과'):
        stages = ['Med-1', 'Med-2']
    elif ',' in s:
        parts = s.split(',')
        for part in parts:
            ps = _parse_year_stage(part.strip())
            if ps:
                stages.append(ps)
    elif s_clean.startswith('대학원'):
        stages = []

    return stages


def load_university_info() -> pd.DataFrame:
    df = pd.read_excel(config.UNIVERSITY_FILE)
    rename = {k: v for k, v in config.UNIVERSITY_COL_RENAME.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)
    df['College'] = df['College'].map(config.COLLEGE_MAP)
    df['Region'] = df['Region'].map(config.REGION_MAP)
    return df


# University feature matrix

def build_university_features(course_df: pd.DataFrame,
                              uni_info: pd.DataFrame) -> pd.DataFrame:
    all_unis = course_df[['College', 'University']].drop_duplicates()
    records = []

    for _, row in all_unis.iterrows():
        uni, college = row['University'], row['College']
        mask = (course_df['University'] == uni) & (course_df['College'] == college)
        udata = course_df[mask]

        total_credits = udata['Credits'].sum()
        total_courses = len(udata)
        cpc = total_credits / total_courses if total_courses > 0 else 0

        domain_cr = {}
        domain_has = {}
        for d in config.DOMAIN_COLS:
            cr = udata[udata[d] == 1]['Credits'].sum()
            domain_cr[f'{d}_Credits'] = cr
            domain_has[f'Has_{d}'] = int(cr > 0)

        breadth = sum(domain_has.values())
        depth = total_credits / breadth if breadth > 0 else 0

        pre_cr = udata[udata['Pre_Medical'] == 1]['Credits'].sum()
        med_cr = udata[udata['Medical'] == 1]['Credits'].sum()
        total_pm = pre_cr + med_cr
        pre_med_ratio = pre_cr / total_pm if total_pm > 0 else 0

        mandatory_ratio = udata['Is_Mandatory_Binary'].mean() if len(udata) > 0 else 0

        cr_vals = [domain_cr[f'{d}_Credits'] for d in config.DOMAIN_COLS]
        total_cat = sum(cr_vals)
        if total_cat > 0:
            props = [c / total_cat for c in cr_vals if c > 0]
            shannon = -sum(p * np.log(p) for p in props)
        else:
            shannon = 0

        ai_core_cr = sum(domain_cr[f'{d}_Credits'] for d in config.AI_CORE_DOMAINS)
        ai_core_ratio = ai_core_cr / total_credits if total_credits > 0 else 0

        has_ai_core = int(any(domain_has[f'Has_{d}'] for d in config.AI_CORE_DOMAINS))
        has_foundational = int(any(domain_has[f'Has_{d}'] for d in config.FOUNDATIONAL_DOMAINS))

        rec = {
            'University': uni,
            'College': college,
            'Total_Credits': total_credits,
            'Total_Courses': total_courses,
            'Credits_per_Course': cpc,
            'Breadth': breadth,
            'Depth': depth,
            'Shannon_Diversity': shannon,
            'Pre_Medical_Ratio': pre_med_ratio,
            'Mandatory_Ratio': mandatory_ratio,
            'AI_Core_Credits': ai_core_cr,
            'AI_Core_Ratio': ai_core_ratio,
            'Has_AI_Core': has_ai_core,
            'Has_Foundational': has_foundational,
            **domain_cr,
            **domain_has,
        }
        records.append(rec)

    df = pd.DataFrame(records)

    df = pd.merge(
        df,
        uni_info[['College', 'University', 'Region', 'Admission_Quota', 'Public_Private']],
        on=['College', 'University'],
        how='left',
    )
    df['Is_Public'] = (df['Public_Private'] == '국립').astype(int)
    df['College'] = pd.Categorical(
        df['College'], categories=config.COLLEGE_ORDER, ordered=True
    )

    return df


# Maturity index

def compute_maturity(uni_df: pd.DataFrame) -> pd.Series:
    levels = []
    for _, row in uni_df.iterrows():
        has_found = row['Has_Foundational']
        has_ai = row['Has_AI_Core']
        n_ai_domains = sum(row[f'Has_{d}'] for d in config.AI_CORE_DOMAINS)
        total_cr = row['Total_Credits']

        if total_cr == 0:
            levels.append(0)
        elif not has_ai:
            levels.append(1)
        elif has_ai and has_found and (n_ai_domains >= 2 and total_cr >= 8):
            levels.append(3)
        else:
            levels.append(2)

    return pd.Series(levels, index=uni_df.index, name='Maturity_Level')


MATURITY_LABELS = {
    0: 'Minimal',
    1: 'Foundational-Only',
    2: 'Intermediate',
    3: 'Advanced',
}


# Convenience loader

def load_all():
    course_df = load_curriculum()
    uni_info = load_university_info()
    uni_df = build_university_features(course_df, uni_info)
    uni_df['Maturity_Level'] = compute_maturity(uni_df)
    uni_df['Maturity_Label'] = uni_df['Maturity_Level'].map(MATURITY_LABELS)

    print(f"Courses: {len(course_df)} (after excluding {len(config.EXCLUDE_COURSES)} non-AI/DS)")
    print(f"Universities: {len(uni_df)}")
    print(f"Maturity distribution: {uni_df['Maturity_Label'].value_counts().to_dict()}")

    return course_df, uni_info, uni_df
