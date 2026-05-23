# Classification loader: frozen course classification + university feature matrix.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config
from data_loader import (
    build_university_features,
    compute_maturity,
    MATURITY_LABELS,
    _parse_year_stage,
)


PROJECT_ROOT = _SCRIPTS_DIR.parent
CONSENSUS_CSV = config.CLASSIFICATION_DIR / "consensus.csv"
M1_CSV = config.CLASSIFICATION_DIR / "m1_rules.csv"
M2_CSV = config.CLASSIFICATION_DIR / "m2_llm.csv"

DOMAINS_SHORT = ["D1", "D2", "D3", "D4", "D5"]
SHORT_TO_FULL = dict(zip(DOMAINS_SHORT, config.DOMAIN_COLS))

STRATEGIES = ("S1_consensus", "S2_single_domain_priority",
              "S3_strict_match", "S4_llm_only", "S5_rule_only",
              "S6_bootstrap")


# Core loaders

def _load_rawdata_curriculum() -> pd.DataFrame:
    df = pd.read_excel(config.CURRICULUM_FILE)
    rename = {k: v for k, v in config.CURRICULUM_COL_RENAME.items()
              if k in df.columns}
    df = df.rename(columns=rename)

    df = df.reset_index(drop=True)
    df["Course_ID"] = df.index + 1

    df["College"] = df["College"].map(config.COLLEGE_MAP)

    df["University"] = df["University"].replace(config.UNIVERSITY_NAME_MAP)
    for (col_en, old_name), new_name in config.UNIVERSITY_NAME_MAP_BY_COLLEGE.items():
        mask = (df["College"] == col_en) & (df["University"] == old_name)
        df.loc[mask, "University"] = new_name

    df["Credits"] = pd.to_numeric(df["Credits"], errors="coerce").fillna(0.0)
    df["Is_Mandatory_Binary"] = (df["Is_Mandatory"] == "필수").astype(int)

    df["Year_Stage"] = df["Year_Raw"].apply(_parse_year_stage)

    return df


def _merge_classifier(raw: pd.DataFrame,
                      classifier_csv: Path,
                      source_label: str) -> pd.DataFrame:
    with open(classifier_csv, "r", encoding="utf-8-sig") as f:
        first = f.readline()
    skip = 1 if first.startswith("#") else 0
    cls = pd.read_csv(classifier_csv, encoding="utf-8-sig", skiprows=skip)

    col_map_m1 = {
        "D1_Quantitative_Foundations": "D1",
        "D2_AI_ML": "D2",
        "D3_Data_Science": "D3",
        "D4_Health_Informatics": "D4",
        "D5_Clinical_AI_Application": "D5",
    }
    for long, short in col_map_m1.items():
        if long in cls.columns and short not in cls.columns:
            cls = cls.rename(columns={long: short})

    keep = ["Course_ID"] + DOMAINS_SHORT
    cls = cls[keep].copy()

    merged = raw.merge(cls, on="Course_ID", how="inner", validate="one_to_one")
    for short, long in SHORT_TO_FULL.items():
        merged[long] = merged[short].astype(int)
    merged = merged.drop(columns=DOMAINS_SHORT)

    merged.attrs["classification_source"] = source_label
    return merged


def load_course_df(classifier: str = "consensus") -> pd.DataFrame:
    raw = _load_rawdata_curriculum()
    if classifier == "consensus":
        return _merge_classifier(raw, CONSENSUS_CSV,
                                 source_label="adjudicated_consensus")
    if classifier == "m1":
        return _merge_classifier(raw, M1_CSV, source_label="rule_based")
    if classifier == "m2":
        return _merge_classifier(raw, M2_CSV, source_label="llm_assistant")
    raise ValueError(f"Unknown classifier: {classifier!r}")


def load_university_info() -> pd.DataFrame:
    df = pd.read_excel(config.UNIVERSITY_FILE)
    rename = {k: v for k, v in config.UNIVERSITY_COL_RENAME.items()
              if k in df.columns}
    df = df.rename(columns=rename)
    df["College"] = df["College"].map(config.COLLEGE_MAP)
    df["Region"] = df["Region"].map(config.REGION_MAP)
    return df


def load_all(classifier: str = "consensus") -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    course_df = load_course_df(classifier=classifier)
    uni_info = load_university_info()
    uni_df = build_university_features(course_df, uni_info)
    uni_df["Maturity_Level"] = compute_maturity(uni_df)
    uni_df["Maturity_Label"] = uni_df["Maturity_Level"].map(MATURITY_LABELS)

    print(f"[classification_loader] classifier={classifier}  "
          f"courses={len(course_df)}  universities={len(uni_df)}")
    print(f"  Maturity distribution: "
          f"{uni_df['Maturity_Label'].value_counts().to_dict()}")

    return course_df, uni_info, uni_df


# Sensitivity reclassifier (for the priority / strict-match strategies)

def reclassify_with_strategy(course_df: pd.DataFrame,
                             strategy: str) -> pd.DataFrame:
    if strategy not in {"single_domain_priority", "strict_match"}:
        raise ValueError(f"Unknown sensitivity strategy: {strategy!r}")

    df = course_df.copy()
    if strategy == "single_domain_priority":
        for _, row in df.iterrows():
            chosen = None
            for d in config.DOMAIN_PRIORITY:
                if row[d] == 1:
                    chosen = d
                    break
            if chosen is None:
                for d in config.DOMAIN_COLS:
                    df.at[row.name, d] = 0
            else:
                for d in config.DOMAIN_COLS:
                    df.at[row.name, d] = 1 if d == chosen else 0
    elif strategy == "strict_match":
        n_domains = df[config.DOMAIN_COLS].sum(axis=1)
        mask = (n_domains != 1)
        for d in config.DOMAIN_COLS:
            df.loc[mask, d] = 0
    return df
