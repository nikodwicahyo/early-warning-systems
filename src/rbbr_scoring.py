"""
RBBR Scoring Module
Functions untuk menghitung skor RBBR berdasarkan threshold OJK
"""

import numpy as np
import pandas as pd
from typing import Union

from .config import RBBRConfig

def score_npl(npl: float) -> int:
    """Convert NPL to score 1-5"""
    if pd.isna(npl):
        return 3

    for pk, (lower, upper) in RBBRConfig.NPL_THRESHOLDS.items():
        if lower <= npl < upper:
            return pk
    return 5

def score_car(car: float) -> int:
    """Convert CAR to score 1-5"""
    if pd.isna(car):
        return 3

    for pk, (lower, upper) in RBBRConfig.CAR_THRESHOLDS.items():
        if lower <= car < upper:
            return pk
    return 5

def score_ldr(ldr: float) -> int:
    """Convert LDR to score 1-5"""
    if pd.isna(ldr):
        return 3

    for pk, threshold in RBBRConfig.LDR_THRESHOLDS.items():
        if isinstance(threshold, tuple):
            if threshold[0] <= ldr <= threshold[1]:
                return pk
        elif isinstance(threshold, list):
            for lower, upper in threshold:
                if lower <= ldr <= upper:
                    return pk
    return 5

def score_roa(roa: float) -> int:
    """Convert ROA to score 1-5"""
    if pd.isna(roa):
        return 3

    for pk, (lower, upper) in RBBRConfig.ROA_THRESHOLDS.items():
        if lower <= roa < upper:
            return pk
    return 5

def score_bopo(bopo: float) -> int:
    """Convert BOPO to score 1-5"""
    if pd.isna(bopo):
        return 3

    for pk, (lower, upper) in RBBRConfig.BOPO_THRESHOLDS.items():
        if lower <= bopo < upper:
            return pk
    return 5

def compute_rbbr_score(row: pd.Series) -> float:
    """
    Compute composite RBBR score

    Args:
        row: DataFrame row with CAR, NPL_gross, LDR, ROA, BOPO

    Returns:
        Composite score (1.0 - 5.0)
    """
    # Risk profile (rata-rata dari NPL, LDR, CAR)
    risk_score = np.mean([
        score_npl(row.get('NPL_gross', 3)),
        score_ldr(row.get('LDR', 3)),
        score_car(row.get('CAR', 3)),
    ])

    # Rentabilitas (profitability)
    rent_score = np.mean([
        score_roa(row.get('ROA', 3)),
        score_bopo(row.get('BOPO', 3)),
    ])

    # Permodalan
    perm_score = score_car(row.get('CAR', 3))

    # GCG (default - bisa diganti dengan data GCG aktual)
    gcg_score = 2.0

    # Weighted composite
    composite = (
        RBBRConfig.WEIGHTS['risk_profile'] * risk_score +
        RBBRConfig.WEIGHTS['gcg'] * gcg_score +
        RBBRConfig.WEIGHTS['rentabilitas'] * rent_score +
        RBBRConfig.WEIGHTS['permodalan'] * perm_score
    )

    return round(composite, 3)

def score_to_pk(score: float) -> int:
    """Convert continuous score to PK (1-5)"""
    if score < 1.5:
        return 1
    elif score < 2.5:
        return 2
    elif score < 3.5:
        return 3
    elif score < 4.5:
        return 4
    else:
        return 5

def pk_to_sinyal(pk: int) -> str:
    """Convert PK to EWS signal"""
    if pk <= 2:
        return 'hijau'
    elif pk == 3:
        return 'kuning'
    else:
        return 'merah'

def batch_score_rbbr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RBBR scores for entire DataFrame

    Args:
        df: DataFrame with required columns

    Returns:
        DataFrame with added columns: rbbr_score, pk_computed, sinyal_ews, and pillar scores
    """
    df = df.copy()

    # Compute individual scores
    df['score_npl'] = df['NPL_gross'].apply(score_npl)
    df['score_car'] = df['CAR'].apply(score_car)
    df['score_ldr'] = df['LDR'].apply(score_ldr)
    df['score_roa'] = df['ROA'].apply(score_roa)
    df['score_bopo'] = df['BOPO'].apply(score_bopo)

    # Compute pillar scores
    df['score_risk_profile'] = df[['score_npl', 'score_ldr', 'score_car']].mean(axis=1)
    df['score_rentabilitas'] = df[['score_roa', 'score_bopo']].mean(axis=1)
    df['score_permodalan'] = df['score_car']
    df['score_gcg'] = 2.0 # Default

    # Compute composite
    df['rbbr_score'] = (
        RBBRConfig.WEIGHTS['risk_profile'] * df['score_risk_profile'] +
        RBBRConfig.WEIGHTS['gcg'] * df['score_gcg'] +
        RBBRConfig.WEIGHTS['rentabilitas'] * df['score_rentabilitas'] +
        RBBRConfig.WEIGHTS['permodalan'] * df['score_permodalan']
    )

    df['pk_computed'] = df['rbbr_score'].apply(score_to_pk)
    df['sinyal_ews'] = df['pk_computed'].apply(pk_to_sinyal)

    return df