"""
Feature Engineering for RBBR Models
Full implementation matching OJK_RBBR_EWS_v4_2.ipynb pipeline:
  - Lag features (lag1, lag2, lag3, lag6)
  - Rolling stats (ma3, ma6, std3, std6, slp3, slp6)
  - Momentum (diff1, diff3, pct_change1)
  - Interaction features (npl_x_bir, nim_x_bir, etc.)
  - Calendar features (month_sin/cos, is_yearend, is_q1, time_idx)
  - Cross-statement derived features (loan_to_asset, equity_ratio, etc.)
  - Multi-step forward targets (fwd1..fwd6)
  - Static categorical encoding (kategori_bank, kepemilikan)
"""

import pandas as pd
import numpy as np
from typing import List, Optional

TARGET_COLS = ['CAR', 'NPL_gross', 'NPL_net', 'ROA', 'ROE', 'NIM', 'BOPO', 'LDR']
MACRO_COLS = ['bi_rate', 'inflasi', 'kurs_usd']

# ============================================================================
# PRIMITIVE HELPERS
# ============================================================================

def _bank_col(df: pd.DataFrame) -> str:
    return 'kode_bank' if 'kode_bank' in df.columns else 'bank'


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    bc = _bank_col(df)
    if bc in df.columns and 'periode' in df.columns:
        return df.sort_values([bc, 'periode'])
    return df


# ============================================================================
# 1. LAG FEATURES (lag1, lag2, lag3, lag6)
# ============================================================================

def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Create lag features: {col}_lag{n} for n in lags."""
    df_feat = _sort(df.copy())
    bc = _bank_col(df_feat)
    actual_lags = lags if lags is not None else [1, 2, 3, 6]

    new_cols = {}
    for col in columns:
        if col not in df_feat.columns:
            continue
        for lag in actual_lags:
            cname = f'{col}_lag{lag}'
            if cname in df_feat.columns:
                continue
            if bc in df_feat.columns:
                new_cols[cname] = df_feat.groupby(bc)[col].shift(lag)
            else:
                new_cols[cname] = df_feat[col].shift(lag)

    if new_cols:
        df_feat = pd.concat([df_feat, pd.DataFrame(new_cols, index=df_feat.index)], axis=1)
    return df_feat


# ============================================================================
# 2. ROLLING FEATURES (ma3, ma6, std3, std6, slp3, slp6)
# ============================================================================

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling OLS slope for a Series."""
    def _slope(vals):
        if len(vals) < 2:
            return np.nan
        x = np.arange(len(vals), dtype=float)
        try:
            return float(np.polyfit(x, vals.astype(float), 1)[0])
        except Exception:
            return np.nan
    return series.rolling(window=window, min_periods=2).apply(_slope, raw=True)


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    Create rolling mean (ma), std, and slope features.
    Naming: {col}_ma{w}, {col}_std{w}, {col}_slp{w}
    """
    df_feat = _sort(df.copy())
    bc = _bank_col(df_feat)
    actual_windows = windows if windows is not None else [3, 6]

    new_cols = {}
    for col in columns:
        if col not in df_feat.columns:
            continue
        for w in actual_windows:
            for stat, suffix in [('mean', f'ma{w}'), ('std', f'std{w}')]:
                cname = f'{col}_{suffix}'
                if cname in df_feat.columns:
                    continue
                if bc in df_feat.columns:
                    series = df_feat.groupby(bc)[col].transform(
                        lambda x, _w=w, _s=stat: getattr(x.rolling(window=_w, min_periods=1), _s)()
                    )
                else:
                    series = getattr(df_feat[col].rolling(window=w, min_periods=1), stat)()
                new_cols[cname] = series
                # Also expose under roll_mean naming for backward compat
                if stat == 'mean':
                    new_cols[f'{col}_roll_mean_{w}'] = series

            # Slope feature
            slp_name = f'{col}_slp{w}'
            if slp_name not in df_feat.columns:
                if bc in df_feat.columns:
                    new_cols[slp_name] = df_feat.groupby(bc)[col].transform(
                        lambda x, _w=w: _rolling_slope(x, _w)
                    )
                else:
                    new_cols[slp_name] = _rolling_slope(df_feat[col], w)

    if new_cols:
        df_feat = pd.concat([df_feat, pd.DataFrame(new_cols, index=df_feat.index)], axis=1)
    return df_feat


# ============================================================================
# 3. MOMENTUM FEATURES (diff1, diff3, pct_change1)
# ============================================================================

def create_momentum_features(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Create diff1, diff3, pct_change1 for each column."""
    df_feat = _sort(df.copy())
    bc = _bank_col(df_feat)

    new_cols = {}
    for col in columns:
        if col not in df_feat.columns:
            continue
        for diff_n, suffix in [(1, 'diff1'), (3, 'diff3')]:
            cname = f'{col}_{suffix}'
            if cname not in df_feat.columns:
                if bc in df_feat.columns:
                    new_cols[cname] = df_feat.groupby(bc)[col].diff(diff_n)
                else:
                    new_cols[cname] = df_feat[col].diff(diff_n)
        pct_name = f'{col}_pct_change1'
        if pct_name not in df_feat.columns:
            if bc in df_feat.columns:
                new_cols[pct_name] = df_feat.groupby(bc)[col].pct_change(1)
            else:
                new_cols[pct_name] = df_feat[col].pct_change(1)

    if new_cols:
        df_feat = pd.concat([df_feat, pd.DataFrame(new_cols, index=df_feat.index)], axis=1)
    return df_feat


# ============================================================================
# 4. INTERACTION FEATURES
# ============================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create macro × rasio interaction features:
      npl_x_bir, nim_x_bir, ldr_x_bir, bopo_x_inf, car_x_kurs,
      credit_cost_x_npl, casa_x_nim, roe_decomp
    """
    df_feat = df.copy()
    new_cols = {}

    def _safe_mul(a, b, fill=0.0):
        sa = df_feat[a] if a in df_feat.columns else pd.Series(fill, index=df_feat.index)
        sb = df_feat[b] if b in df_feat.columns else pd.Series(fill, index=df_feat.index)
        return (sa * sb).fillna(fill)

    pairs = [
        ('NPL_gross', 'bi_rate',   'npl_x_bir'),
        ('NIM',       'bi_rate',   'nim_x_bir'),
        ('LDR',       'bi_rate',   'ldr_x_bir'),
        ('BOPO',      'inflasi',   'bopo_x_inf'),
        ('CAR',       'kurs_usd',  'car_x_kurs'),
    ]
    for col_a, col_b, out_name in pairs:
        if out_name not in df_feat.columns:
            new_cols[out_name] = _safe_mul(col_a, col_b)

    # credit_cost_x_npl: credit_cost_ratio × NPL_gross
    if 'credit_cost_ratio' in df_feat.columns and 'NPL_gross' in df_feat.columns:
        if 'credit_cost_x_npl' not in df_feat.columns:
            new_cols['credit_cost_x_npl'] = (df_feat['credit_cost_ratio'] * df_feat['NPL_gross']).fillna(0)

    # casa_x_nim: casa_ratio × NIM
    if 'casa_ratio' in df_feat.columns and 'NIM' in df_feat.columns:
        if 'casa_x_nim' not in df_feat.columns:
            new_cols['casa_x_nim'] = (df_feat['casa_ratio'] * df_feat['NIM']).fillna(0)

    # roe_decomp: ROA × (1 / CAR proxy)  → ROE decomposition signal
    if 'ROA' in df_feat.columns and 'CAR' in df_feat.columns:
        if 'roe_decomp' not in df_feat.columns:
            new_cols['roe_decomp'] = (df_feat['ROA'] / df_feat['CAR'].replace(0, np.nan)).fillna(0)

    if new_cols:
        df_feat = pd.concat([df_feat, pd.DataFrame(new_cols, index=df_feat.index)], axis=1)
    return df_feat


# ============================================================================
# 5. CALENDAR FEATURES
# ============================================================================

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add: month, quarter, is_yearend, is_q1, is_mid_year,
         month_sin, month_cos, time_idx (per bank)
    """
    df_feat = df.copy()
    bc = _bank_col(df_feat)

    if 'periode' not in df_feat.columns:
        return df_feat

    df_feat['periode'] = pd.to_datetime(df_feat['periode'])
    new_cols = {}

    if 'month' not in df_feat.columns:
        new_cols['month'] = df_feat['periode'].dt.month
    if 'quarter' not in df_feat.columns:
        new_cols['quarter'] = df_feat['periode'].dt.quarter
    if 'is_yearend' not in df_feat.columns:
        new_cols['is_yearend'] = (df_feat['periode'].dt.month == 12).astype(int)
    if 'is_q1' not in df_feat.columns:
        new_cols['is_q1'] = (df_feat['periode'].dt.month <= 3).astype(int)
    if 'is_mid_year' not in df_feat.columns:
        new_cols['is_mid_year'] = df_feat['periode'].dt.month.isin([6, 7]).astype(int)
    if 'month_sin' not in df_feat.columns:
        m = df_feat['periode'].dt.month
        new_cols['month_sin'] = np.sin(2 * np.pi * m / 12)
        new_cols['month_cos'] = np.cos(2 * np.pi * m / 12)

    # time_idx — monotonically increasing integer per bank group
    if 'time_idx' not in df_feat.columns:
        if bc in df_feat.columns:
            new_cols['time_idx'] = df_feat.groupby(bc).cumcount()
        else:
            new_cols['time_idx'] = np.arange(len(df_feat))

    if new_cols:
        df_feat = pd.concat([df_feat, pd.DataFrame(new_cols, index=df_feat.index)], axis=1)
    return df_feat


# ============================================================================
# 6. CROSS-STATEMENT DERIVED FEATURES
# ============================================================================

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build cross-statement derived features that capture bank quality signals:
      loan_to_asset, equity_ratio, leverage_ratio,
      dpk_total, current_account_ratio, saving_ratio, deposit_ratio, casa_ratio,
      cost_of_fund, net_interest_margin_proxy, credit_cost_ratio,
      fee_to_income, komitmen_to_aset, off_balance_ratio, ckpn_coverage_est
    """
    df_feat = df.copy()
    new_cols = {}

    def _col(name, default=np.nan):
        return df_feat[name] if name in df_feat.columns else pd.Series(default, index=df_feat.index)

    total_aset = _col('total_aset')
    kredit = _col('kredit_diberikan').combine_first(_col('kredit_yang_diberikan'))
    modal = _col('modal_inti').combine_first(_col('ekuitas'))
    dpk = _col('dpk').combine_first(_col('dana_pihak_ketiga'))
    tabungan = _col('tabungan')
    giro = _col('giro')
    deposito = _col('deposito')
    nii = _col('NII').combine_first(_col('pendapatan_bunga_bersih'))
    biaya_bunga = _col('biaya_bunga')
    biaya_prov = _col('biaya_provisi')
    pendapatan_fee = _col('pendapatan_fee').combine_first(_col('fee_income'))
    pendapatan_total = _col('pendapatan_total').combine_first(_col('total_revenue'))
    komitmen = _col('tagihan_komitmen').combine_first(_col('komitmen'))
    off_balance = _col('off_balance_sheet').combine_first(komitmen)
    ckpn = _col('ckpn').combine_first(_col('cadangan_kerugian'))

    safe = lambda num, den: num.divide(den.replace(0, np.nan)).fillna(0)

    if 'loan_to_asset' not in df_feat.columns:
        new_cols['loan_to_asset'] = safe(kredit, total_aset)
    if 'equity_ratio' not in df_feat.columns:
        new_cols['equity_ratio'] = safe(modal, total_aset)
    if 'leverage_ratio' not in df_feat.columns:
        new_cols['leverage_ratio'] = safe(total_aset, modal)

    if 'dpk_total' not in df_feat.columns and not dpk.isna().all():
        new_cols['dpk_total'] = dpk

    if not dpk.isna().all():
        if 'current_account_ratio' not in df_feat.columns and not giro.isna().all():
            new_cols['current_account_ratio'] = safe(giro, dpk)
        if 'saving_ratio' not in df_feat.columns and not tabungan.isna().all():
            new_cols['saving_ratio'] = safe(tabungan, dpk)
        if 'deposit_ratio' not in df_feat.columns and not deposito.isna().all():
            new_cols['deposit_ratio'] = safe(deposito, dpk)
        if 'casa_ratio' not in df_feat.columns:
            casa = (giro.fillna(0) + tabungan.fillna(0))
            new_cols['casa_ratio'] = safe(casa, dpk)

    if 'cost_of_fund' not in df_feat.columns and not biaya_bunga.isna().all():
        new_cols['cost_of_fund'] = safe(biaya_bunga, dpk)
    if 'net_interest_margin_proxy' not in df_feat.columns:
        new_cols['net_interest_margin_proxy'] = safe(nii, total_aset)
    if 'credit_cost_ratio' not in df_feat.columns and not biaya_prov.isna().all():
        new_cols['credit_cost_ratio'] = safe(biaya_prov, kredit)
    if 'fee_to_income' not in df_feat.columns and not pendapatan_fee.isna().all():
        new_cols['fee_to_income'] = safe(pendapatan_fee, pendapatan_total)
    if 'komitmen_to_aset' not in df_feat.columns:
        new_cols['komitmen_to_aset'] = safe(komitmen, total_aset)
    if 'off_balance_ratio' not in df_feat.columns:
        new_cols['off_balance_ratio'] = safe(off_balance, total_aset)
    if 'ckpn_coverage_est' not in df_feat.columns and not ckpn.isna().all():
        npl_abs = kredit * _col('NPL_gross').fillna(0) / 100
        new_cols['ckpn_coverage_est'] = safe(ckpn, npl_abs)

    if new_cols:
        df_feat = pd.concat([df_feat, pd.DataFrame(new_cols, index=df_feat.index)], axis=1)
    return df_feat


# ============================================================================
# 7. FORWARD TARGET LABELS (fwd1..fwd6)
# ============================================================================

def create_forward_targets(
    df: pd.DataFrame,
    targets: List[str] = None,
    horizons: List[int] = None,
) -> pd.DataFrame:
    """
    Create forward-shifted target columns: {col}_fwd{h}
    Used as training labels for multi-step prediction.
    """
    df_feat = _sort(df.copy())
    bc = _bank_col(df_feat)
    actual_targets = targets if targets is not None else TARGET_COLS + ['rbbr_score']
    actual_horizons = horizons if horizons is not None else list(range(1, 7))

    new_cols = {}
    for col in actual_targets:
        if col not in df_feat.columns:
            continue
        for h in actual_horizons:
            cname = f'{col}_fwd{h}'
            if cname not in df_feat.columns:
                if bc in df_feat.columns:
                    new_cols[cname] = df_feat.groupby(bc)[col].shift(-h)
                else:
                    new_cols[cname] = df_feat[col].shift(-h)

    if new_cols:
        df_feat = pd.concat([df_feat, pd.DataFrame(new_cols, index=df_feat.index)], axis=1)
    return df_feat


# ============================================================================
# 8. STATIC CATEGORICAL ENCODING
# ============================================================================

def encode_static_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode kategori_bank and kepemilikan."""
    df_feat = df.copy()
    for col in ['kategori_bank', 'kepemilikan']:
        if col in df_feat.columns:
            dummies = pd.get_dummies(df_feat[col], prefix=col, drop_first=False)
            # Avoid duplicating columns
            dummies = dummies[[c for c in dummies.columns if c not in df_feat.columns]]
            df_feat = pd.concat([df_feat, dummies], axis=1)
    return df_feat


# ============================================================================
# 9. MASTER FUNCTION: create_full_feature_set
# ============================================================================

def create_full_feature_set(
    df: pd.DataFrame,
    include_forward_targets: bool = True,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline matching OJK_RBBR_EWS_v4_2.ipynb:
      1. Derived cross-statement features
      2. Lag features (1, 2, 3, 6)
      3. Rolling stats (ma, std, slp for windows 3, 6)
      4. Momentum (diff1, diff3, pct_change1)
      5. Interaction features (macro × rasio)
      6. Calendar features
      7. Forward targets (fwd1..fwd6)  [optional]
      8. Static categorical encoding
    Returns enriched DataFrame.
    """
    df_out = df.copy()
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    # Step 1: Derived features (cross-statement)
    df_out = create_derived_features(df_out)

    # Step 2: Lag features for main targets + macro + derived
    lag_cols = [c for c in TARGET_COLS + MACRO_COLS + [
        'loan_to_asset', 'equity_ratio', 'casa_ratio', 'credit_cost_ratio',
        'net_interest_margin_proxy', 'rbbr_score',
    ] if c in df_out.columns]
    df_out = create_lag_features(df_out, columns=lag_cols, lags=[1, 2, 3, 6])

    # Step 3: Rolling stats
    roll_cols = [c for c in TARGET_COLS + MACRO_COLS if c in df_out.columns]
    df_out = create_rolling_features(df_out, columns=roll_cols, windows=[3, 6])

    # Step 4: Momentum
    mom_cols = [c for c in TARGET_COLS if c in df_out.columns]
    df_out = create_momentum_features(df_out, columns=mom_cols)

    # Step 5: Interactions
    df_out = create_interaction_features(df_out)

    # Step 6: Calendar
    df_out = create_calendar_features(df_out)

    # Step 7: Forward targets
    if include_forward_targets:
        fwd_base = [c for c in TARGET_COLS + ['rbbr_score'] if c in df_out.columns]
        df_out = create_forward_targets(df_out, targets=fwd_base)

    # Step 8: Categorical encoding
    df_out = encode_static_categoricals(df_out)

    # Deduplicate columns
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    return df_out
