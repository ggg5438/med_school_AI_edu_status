# Central configuration: paths, mappings, domain definitions, palettes.

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAWDATA_DIR = DATA_DIR / 'raw'
CLASSIFICATION_DIR = DATA_DIR / 'classification'
CURRICULUM_FILE = RAWDATA_DIR / '교육과정현황조사 최종본.xlsx'
UNIVERSITY_FILE = RAWDATA_DIR / '대학정보.xlsx'

RESULTS_DIR = PROJECT_ROOT / 'results'
STATISTICS_DIR = RESULTS_DIR / 'statistics'
FIGURES_DIR = PROJECT_ROOT / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

# Column renaming: Korean → English
CURRICULUM_COL_RENAME = {
    '학교구분': 'College',
    '학교명': 'University',
    '과목명': 'Course_Name',
    '학점': 'Credits',
    '필수': 'Is_Mandatory',
    '예과과목': 'Pre_Medical',
    '본과과목': 'Medical',
    '학년': 'Year_Raw',
}

UNIVERSITY_COL_RENAME = {
    '학교구분': 'College',
    '학교명': 'University',
    '지역': 'Region',
    '입학정원': 'Admission_Quota',
    '국립사립': 'Public_Private',
}

# Value mappings
COLLEGE_MAP = {
    '의대': 'Medicine',
    '치대': 'Dentistry',
    '한의대': 'Korean Medicine',
}
COLLEGE_ORDER = ['Medicine', 'Dentistry', 'Korean Medicine']

REGION_MAP = {
    '서울경기인천': 'Seoul/Gyeonggi',
    '강원권': 'Gangwon',
    '충청권': 'Chungcheong',
    '전라제주권': 'Jeolla/Jeju',
    '경남권': 'Gyeongnam',
    '경북권': 'Gyeongbuk',
}

YEAR_STAGES = ['Pre-1', 'Pre-2', 'Med-1', 'Med-2', 'Med-3', 'Med-4']

# Domain classification (synthesized from 3 frameworks)
DOMAIN_COLS = [
    'D1_Quantitative_Foundations',
    'D2_AI_ML',
    'D3_Data_Science',
    'D4_Health_Informatics',
    'D5_Clinical_AI_Application',
]

DOMAIN_LABELS = {
    'D1_Quantitative_Foundations': 'Quantitative Foundations',
    'D2_AI_ML': 'AI & Machine Learning',
    'D3_Data_Science': 'Data Science',
    'D4_Health_Informatics': 'Health Informatics',
    'D5_Clinical_AI_Application': 'Clinical AI Application',
}

DOMAIN_SHORT = {
    'D1_Quantitative_Foundations': 'Quant',
    'D2_AI_ML': 'AI/ML',
    'D3_Data_Science': 'DS',
    'D4_Health_Informatics': 'HI',
    'D5_Clinical_AI_Application': 'ClinAI',
}

DOMAIN_KEYWORDS = {
    'D1_Quantitative_Foundations': [
        '통계', '확률', '수학', '수리',
        '프로그래밍', '코딩', '코드', '파이썬', 'R프로그래밍',
        '컴퓨팅사고', '컴퓨터사고', 'SW', '소프트웨어',
        '컴퓨터과학', '컴퓨터코딩', '컴퓨터 활용',
        '컴퓨팅', '알고리즘',
    ],
    'D2_AI_ML': [
        '인공지능', 'AI', 'ai', '머신러닝', '딥러닝',
        '기계학습', '신경망', '로봇',
    ],
    'D3_Data_Science': [
        '데이터과학', '데이터사이언스',
        '빅데이터', '데이터분석', '데이터리터러시',
        '데이터', '자료처리', '자료 분석',
        '데이터 과학', '데이터 사이언스',
    ],
    'D4_Health_Informatics': [
        '정보학', '정보기술', '정보 기술', '정보와 기술',
        '의료정보', '한의정보', '생물정보',
        '디지털', '융합정보',
        'EBM', '정보의학',
    ],
    'D5_Clinical_AI_Application': [
        '의료AI', '의료인공지능', '의료 인공지능',
        '치과의료', '치의학인공지능',
        '의료영상분석', '의료와 인공지능',
        '의료의미래', '미래의학',
        '의료빅데이터', '의료와데이터',
        '의료데이터', '의료딥러닝',
        '의료와 데이터',
        '한의학과인공지능', '한의학DB',
        '의과학',
    ],
}

EXCLUDE_COURSES = {
    '(EAL-A)삶과 4차 산업혁명',
    'x',
    'ai시대쓰기의힘',
}

TYPO_CORRECTIONS = {
    '정보과 기술': '정보와 기술',
    '통개학개론': '통계학개론',
}

UNIVERSITY_NAME_MAP = {
    '차의과대학교': '차의과학대학교',
}
UNIVERSITY_NAME_MAP_BY_COLLEGE = {
    ('Dentistry', '단국대학교'): '단국대학교 글로컬캠퍼스',
}

MANUAL_OVERRIDES = {
    '치과의료통계학및실습': {'D5_Clinical_AI_Application': 0},
}

AI_CORE_DOMAINS = ['D2_AI_ML', 'D3_Data_Science', 'D5_Clinical_AI_Application']
FOUNDATIONAL_DOMAINS = ['D1_Quantitative_Foundations']
INTERMEDIARY_DOMAINS = ['D4_Health_Informatics']

# Maturity analysis
MATURITY_ORDER = ['Minimal', 'Foundational-Only', 'Intermediate', 'Advanced']

MATURITY_COLORS = {
    'Minimal': '#D4D4D4',
    'Foundational-Only': '#D65F5F',
    'Intermediate': '#4878D0',
    'Advanced': '#6ACC64',
}

# Sensitivity analysis — domain priority for single-domain assignment
DOMAIN_PRIORITY = [
    'D5_Clinical_AI_Application',
    'D2_AI_ML',
    'D3_Data_Science',
    'D4_Health_Informatics',
    'D1_Quantitative_Foundations',
]

DATA_COLLECTION_PERIOD = "April to June 2025"

# Statistical constants
ALPHA = 0.05
RANDOM_STATE = 42

# Color palettes — colorblind-safe
COLLEGE_COLORS = {
    'Medicine': '#4878D0',
    'Dentistry': '#EE854A',
    'Korean Medicine': '#6ACC64',
}
COLLEGE_PALETTE = [COLLEGE_COLORS[c] for c in COLLEGE_ORDER]

DOMAIN_COLORS = {
    'D1_Quantitative_Foundations': '#D65F5F',
    'D2_AI_ML': '#4878D0',
    'D3_Data_Science': '#956CB4',
    'D4_Health_Informatics': '#8C613C',
    'D5_Clinical_AI_Application': '#DC7EC0',
}
DOMAIN_PALETTE = [DOMAIN_COLORS[d] for d in DOMAIN_COLS]

PP_COLORS = {'Public': '#4878D0', 'Private': '#D4D4D4'}

# Figure sizes — Nature column standards
SINGLE_COL = 89 / 25.4
ONE_HALF_COL = 136 / 25.4
DOUBLE_COL = 183 / 25.4
FIG_DPI = 600
