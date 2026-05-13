"""
Data Augmentation Module
Matches Section 10 of OJK_RBBR_EWS_v4_2.ipynb:
  10.1  Jitter AR(1)
  10.2  Stress scenario
  10.3  Recovery scenario
  10.4  Recompute RBBR/PK/EWS labels after augmentation

Usage:
    from src.models.data_augmentation import augment_and_recompute
    df_aug = augment_and_recompute(df)
"""

import numpy as np
import pandas as pd
from typing import Optional

TARGET_COLS = ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR']

# Realistic bounds per column — keep synthetic data plausible
_BOUNDS = {
    'CAR':      (8.0,  50.0),
    'NPL_gross': (0.1, 30.0),
    'NPL_net':  (0.0,  25.0),
    'ROA':      (-3.0, 6.0),
    'ROE':      (-20.0, 35.0),
    'NIM':      (0.5,  12.0),
    'BOPO':     (40.0, 120.0),
    'LDR':      (30.0, 150.0),
    'bi_rate':  (1.0,  15.0),
    'inflasi':  (-2.0, 20.0),
    'kurs_usd': (10000.0, 25000.0),
}


def _clip(df: pd.DataFrame) -> pd.DataFrame:
    """Clip all columns to their realistic bounds."""
    for col, (lo, hi) in _BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


# ============================================================================
# 10.1  AR(1) Jitter
# ============================================================================

def augment_jitter_ar1(
    df: pd.DataFrame,
    noise_scale: float = 0.02,
    ar_phi: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add realistic AR(1) noise to financial ratios.
    noise_t = phi * noise_{t-1} + epsilon_t,  epsilon ~ N(0, scale * std(col))
    """
    rng = np.random.default_rng(seed)
    df_aug = df.copy()

    num_cols = [c for c in TARGET_COLS if c in df_aug.columns]
    for col in num_cols:
        sigma = df_aug[col].std() * noise_scale
        if sigma == 0 or np.isnan(sigma):
            continue
        n = len(df_aug)
        eps = rng.normal(0, sigma, n)
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = ar_phi * noise[i - 1] + eps[i]
        df_aug[col] = (df_aug[col] + noise).clip(
            *_BOUNDS.get(col, (-np.inf, np.inf))
        )

    df_aug = _clip(df_aug)
    return df_aug


# ============================================================================
# 10.2  Stress Scenario
# ============================================================================

def augment_stress_scenario(
    df: pd.DataFrame,
    intensity: float = 1.0,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simulate a stress scenario:
      - NPL_gross  ↑ (deteriorates)
      - CAR        ↓
      - ROA        ↓
      - BOPO       ↑
      - LDR        slight ↓ (credit contraction)
      - NIM        ↓
      - kredit_diberikan  ↓ (if available)
      - laba_bersih       ↓ (if available)
    """
    rng = np.random.default_rng(seed)
    df_aug = df.copy()

    # Drift magnitudes (fraction of column std, scaled by intensity)
    stress_map = {
        'NPL_gross':  +0.3 * intensity,
        'NPL_net':    +0.2 * intensity,
        'CAR':        -0.15 * intensity,
        'ROA':        -0.20 * intensity,
        'ROE':        -0.25 * intensity,
        'BOPO':       +0.25 * intensity,
        'NIM':        -0.15 * intensity,
        'LDR':        -0.10 * intensity,
    }
    balance_map = {
        'kredit_diberikan':  -0.10 * intensity,
        'kredit_yang_diberikan': -0.10 * intensity,
        'laba_bersih':       -0.30 * intensity,
        'pendapatan_total':  -0.10 * intensity,
        'total_aset':        -0.05 * intensity,
    }

    for col, drift_frac in {**stress_map, **balance_map}.items():
        if col not in df_aug.columns:
            continue
        sigma = df_aug[col].std()
        if sigma == 0 or np.isnan(sigma):
            continue
        drift = drift_frac * sigma
        noise = rng.normal(drift, abs(drift) * 0.2, len(df_aug))
        df_aug[col] = df_aug[col] + noise

    df_aug = _clip(df_aug)
    return df_aug


# ============================================================================
# 10.3  Recovery Scenario
# ============================================================================

def augment_recovery_scenario(
    df: pd.DataFrame,
    intensity: float = 0.8,
    seed: int = 456,
) -> pd.DataFrame:
    """
    Simulate post-stress recovery:
    Inverse of stress with slightly lower magnitude (recovery is gradual).
    """
    rng = np.random.default_rng(seed)
    df_aug = df.copy()

    recovery_map = {
        'NPL_gross':  -0.2 * intensity,
        'NPL_net':    -0.15 * intensity,
        'CAR':        +0.12 * intensity,
        'ROA':        +0.15 * intensity,
        'ROE':        +0.20 * intensity,
        'BOPO':       -0.20 * intensity,
        'NIM':        +0.10 * intensity,
        'LDR':        +0.08 * intensity,
    }
    balance_map = {
        'kredit_diberikan':  +0.08 * intensity,
        'kredit_yang_diberikan': +0.08 * intensity,
        'laba_bersih':       +0.20 * intensity,
        'pendapatan_total':  +0.08 * intensity,
        'total_aset':        +0.04 * intensity,
    }

    for col, drift_frac in {**recovery_map, **balance_map}.items():
        if col not in df_aug.columns:
            continue
        sigma = df_aug[col].std()
        if sigma == 0 or np.isnan(sigma):
            continue
        drift = drift_frac * sigma
        noise = rng.normal(drift, abs(drift) * 0.2, len(df_aug))
        df_aug[col] = df_aug[col] + noise

    df_aug = _clip(df_aug)
    return df_aug


# ============================================================================
# 10.4  Recompute RBBR / PK / EWS / Forward targets after augmentation
# ============================================================================

def _recompute_rbbr(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute rbbr_score, pk_computed, sinyal_ews after augmentation."""
    try:
        from src.rbbr_scoring import batch_score_rbbr
        df = batch_score_rbbr(df)
    except Exception:
        pass
    return df


def _recompute_forward_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute forward target columns after augmentation."""
    try:
        from src.models.feature_engineering import create_forward_targets
        fwd_base = [c for c in TARGET_COLS + ['rbbr_score'] if c in df.columns]
        df = create_forward_targets(df, targets=fwd_base)
    except Exception:
        pass
    return df


def _add_ews_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pk_fwd6 and ews_alarm from forward targets."""
    if 'rbbr_score_fwd6' in df.columns and 'pk_fwd6' not in df.columns:
        try:
            from src.rbbr_scoring import score_to_pk, pk_to_sinyal
            df['pk_fwd6'] = df['rbbr_score_fwd6'].apply(score_to_pk)
            df['ews_alarm'] = (
                (df['pk_fwd6'] >= 4) |
                (df['pk_fwd6'] > df.get('pk_computed', 3))
            ).astype(int)
        except Exception:
            pass
    return df


# ============================================================================
# MASTER FUNCTION
# ============================================================================

def augment_and_recompute(
    df: pd.DataFrame,
    max_multiplier: int = 3,
) -> pd.DataFrame:
    """
    Apply all three augmentation strategies then recompute labels.
    Returns a DataFrame up to max_multiplier × original size.

    Args:
        df: Input DataFrame (already feature-engineered is fine)
        max_multiplier: Maximum size relative to original (default 3×)

    Returns:
        Augmented DataFrame with recomputed RBBR/PK/EWS/forward targets
    """
    original = df.copy()
    augmented_parts = [original]

    # AR(1) jitter copy
    try:
        df_jitter = augment_jitter_ar1(original)
        augmented_parts.append(df_jitter)
    except Exception:
        pass

    # Stress scenario copy
    try:
        df_stress = augment_stress_scenario(original)
        augmented_parts.append(df_stress)
    except Exception:
        pass

    # Recovery scenario copy — only add if not exceeding multiplier
    if len(augmented_parts) < max_multiplier:
        try:
            df_recovery = augment_recovery_scenario(original)
            augmented_parts.append(df_recovery)
        except Exception:
            pass

    # Combine
    df_aug = pd.concat(augmented_parts, ignore_index=True)

    # Recompute derived scores and targets on augmented data
    df_aug = _recompute_rbbr(df_aug)
    df_aug = _recompute_forward_targets(df_aug)
    df_aug = _add_ews_labels(df_aug)

    # Deduplicate columns
    df_aug = df_aug.loc[:, ~df_aug.columns.duplicated()]

    return df_aug
