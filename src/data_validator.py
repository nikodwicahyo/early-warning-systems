"""
Data Validation Module
Validates uploaded data for security, continuity, and completeness before retraining
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates uploaded data for retraining"""
    
    REQUIRED_COLS = ['periode', 'kode_bank', 'bank', 'CAR', 'NPL_gross', 'LDR', 'ROA', 'BOPO']
    NUMERIC_COLS = ['CAR', 'NPL_gross', 'NPL_net', 'LDR', 'ROA', 'ROE', 'NIM', 'BOPO', 
                    'bi_rate', 'inflasi', 'kurs_usd']
    
    # Reasonable ranges for financial metrics (min, max)
    RANGES = {
        'CAR': (0, 200),
        'NPL_gross': (0, 100),
        'NPL_net': (0, 100),
        'LDR': (0, 200),
        'ROA': (-50, 50),
        'ROE': (-100, 100),
        'NIM': (-20, 50),
        'BOPO': (0, 500),
        'bi_rate': (0, 50),
        'inflasi': (-20, 100),
        'kurs_usd': (1000, 100000),
    }
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_upload(self, df: pd.DataFrame, existing_data: pd.DataFrame = None) -> Tuple[bool, Dict]:
        """
        Comprehensive validation of uploaded data
        
        Args:
            df: Uploaded dataframe
            existing_data: Existing historical data for continuity check
            
        Returns:
            (is_valid, validation_report)
        """
        self.errors = []
        self.warnings = []
        
        # 1. Basic structure validation
        self._validate_structure(df)
        
        # 2. Data type validation
        self._validate_datatypes(df)
        
        # 3. Range validation
        self._validate_ranges(df)
        
        # 4. Continuity validation (if existing data provided)
        if existing_data is not None and not existing_data.empty:
            self._validate_continuity(df, existing_data)
        
        # 5. Completeness validation
        self._validate_completeness(df)
        
        # 6. Security checks
        self._validate_security(df)
        
        is_valid = len(self.errors) == 0
        
        report = {
            'is_valid': is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'summary': {
                'total_rows': len(df),
                'unique_banks': df['kode_bank'].nunique() if 'kode_bank' in df.columns else 0,
                'date_range': self._get_date_range(df),
                'error_count': len(self.errors),
                'warning_count': len(self.warnings),
            }
        }
        
        return is_valid, report
    
    def _validate_structure(self, df: pd.DataFrame):
        """Check required columns exist"""
        if df.empty:
            self.errors.append("Data kosong - tidak ada baris data")
            return
        
        missing_cols = [col for col in self.REQUIRED_COLS if col not in df.columns]
        if missing_cols:
            self.errors.append(f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}")
    
    def _validate_datatypes(self, df: pd.DataFrame):
        """Validate and convert data types"""
        if 'periode' not in df.columns:
            return
        
        # Check periode can be converted to datetime
        try:
            pd.to_datetime(df['periode'])
        except Exception as e:
            self.errors.append(f"Kolom 'periode' tidak valid sebagai tanggal: {str(e)}")
        
        # Check numeric columns
        for col in self.NUMERIC_COLS:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    null_count = pd.to_numeric(df[col], errors='coerce').isnull().sum()
                    if null_count > 0:
                        self.warnings.append(f"Kolom '{col}' memiliki {null_count} nilai tidak valid")
                except Exception as e:
                    self.errors.append(f"Kolom '{col}' tidak dapat dikonversi ke numerik: {str(e)}")
    
    def _validate_ranges(self, df: pd.DataFrame):
        """Check if values are within reasonable ranges"""
        for col, (min_val, max_val) in self.RANGES.items():
            if col not in df.columns:
                continue
            
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                out_of_range = ((numeric_col < min_val) | (numeric_col > max_val)).sum()
                
                if out_of_range > 0:
                    self.warnings.append(
                        f"Kolom '{col}': {out_of_range} nilai di luar rentang wajar ({min_val}-{max_val})"
                    )
            except Exception:
                pass
    
    def _validate_continuity(self, df: pd.DataFrame, existing_data: pd.DataFrame):
        """
        Validate that new data continues from existing data without gaps
        Critical for time series models
        """
        if 'periode' not in df.columns or 'periode' not in existing_data.columns:
            self.warnings.append("Tidak dapat memvalidasi kontinuitas - kolom 'periode' tidak ditemukan")
            return
        
        try:
            # Convert to datetime
            new_periods = pd.to_datetime(df['periode'])
            existing_periods = pd.to_datetime(existing_data['periode'])
            
            # Get latest date from existing data
            latest_existing = existing_periods.max()
            earliest_new = new_periods.min()
            
            # Check if new data starts after existing data
            latest_new = new_periods.max()
            if earliest_new <= latest_existing and latest_new <= latest_existing:
                self.warnings.append(
                    f"Data baru dimulai dari {earliest_new.strftime('%Y-%m')} "
                    f"tetapi data existing sudah ada sampai {latest_existing.strftime('%Y-%m')}. "
                    f"Data duplikat akan ditimpa."
                )
            
            # Check for gaps (more than 2 months)
            gap_months = (earliest_new.year - latest_existing.year) * 12 + (earliest_new.month - latest_existing.month)
            
            if gap_months > 2:
                self.errors.append(
                    f"Gap data terlalu besar: {gap_months} bulan antara data existing "
                    f"({latest_existing.strftime('%Y-%m')}) dan data baru ({earliest_new.strftime('%Y-%m')}). "
                    f"Data harus kontinyu untuk model time series."
                )
            elif gap_months == 2:
                self.warnings.append(
                    f"Ada gap 1 bulan antara data existing dan data baru. "
                    f"Disarankan untuk melengkapi data bulan {(latest_existing + timedelta(days=31)).strftime('%Y-%m')}"
                )
            
            # Check bank continuity
            if 'kode_bank' in df.columns and 'kode_bank' in existing_data.columns:
                existing_banks = set(existing_data['kode_bank'].unique())
                new_banks = set(df['kode_bank'].unique())
                
                missing_banks = existing_banks - new_banks
                if missing_banks and len(missing_banks) < len(existing_banks):
                    self.warnings.append(
                        f"Data baru tidak mencakup {len(missing_banks)} bank yang ada di data existing: "
                        f"{', '.join(list(missing_banks)[:5])}{'...' if len(missing_banks) > 5 else ''}"
                    )
        
        except Exception as e:
            self.warnings.append(f"Error saat validasi kontinuitas: {str(e)}")
    
    def _validate_completeness(self, df: pd.DataFrame):
        """Check data completeness"""
        if df.empty:
            return
        
        # Check for missing values in critical columns
        critical_cols = ['periode', 'kode_bank', 'CAR', 'NPL_gross', 'ROA']
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                
                if null_pct > 50:
                    self.errors.append(f"Kolom '{col}' memiliki terlalu banyak nilai kosong ({null_pct:.1f}%)")
                elif null_pct > 10:
                    self.warnings.append(f"Kolom '{col}' memiliki {null_pct:.1f}% nilai kosong")
        
        # Check if we have enough data points per bank
        if 'kode_bank' in df.columns and 'periode' in df.columns:
            bank_counts = df.groupby('kode_bank').size()
            insufficient_banks = bank_counts[bank_counts < 3].index.tolist()
            
            if insufficient_banks:
                self.warnings.append(
                    f"{len(insufficient_banks)} bank memiliki kurang dari 3 periode data: "
                    f"{', '.join(insufficient_banks[:5])}{'...' if len(insufficient_banks) > 5 else ''}"
                )
    
    def _validate_security(self, df: pd.DataFrame):
        """Security checks for malicious data"""
        # Check for suspiciously large file
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_mb > 100:
            self.warnings.append(f"File sangat besar ({memory_mb:.1f} MB) - pastikan ini adalah data yang benar")
        
        # Check for SQL injection patterns in string columns
        sql_patterns = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'SELECT', '--', ';', 'UNION']
        for col in df.select_dtypes(include=['object']).columns:
            for pattern in sql_patterns:
                if df[col].astype(str).str.contains(pattern, case=False, na=False).any():
                    self.errors.append(f"Pola mencurigakan ditemukan di kolom '{col}' - file ditolak")
                    return
        
        # Check for extremely long strings (potential buffer overflow)
        for col in df.select_dtypes(include=['object']).columns:
            max_len = df[col].astype(str).str.len().max()
            if max_len > 1000:
                self.warnings.append(f"Kolom '{col}' memiliki nilai sangat panjang ({max_len} karakter)")
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Get date range string"""
        if 'periode' not in df.columns:
            return "N/A"
        
        try:
            periods = pd.to_datetime(df['periode'])
            return f"{periods.min().strftime('%Y-%m')} s/d {periods.max().strftime('%Y-%m')}"
        except Exception:
            return "Invalid"
    
    def check_retraining_readiness(self, data_dir: Path) -> Tuple[bool, Dict]:
        """
        Check if system is ready for retraining.
        Readiness is based on master_panel existence and content.
        Uploads are optional: if present they are validated and shown as info,
        but their absence does NOT block retraining (uploads are cleared after
        each successful retraining cycle).
        """
        errors = []
        warnings = []
        master_file_exists = False
        has_new_upload = False
        newest_file_str = None
        newest_data_period_str = None

        # ── Master panel check (required) ─────────────────────────────────────
        master_file = data_dir / 'processed' / 'master_panel.csv'
        if not master_file.exists():
            master_file = data_dir / 'processed' / 'master_panel.parquet'

        master_file_exists = master_file.exists()

        if not master_file_exists:
            errors.append("File master data tidak ditemukan (master_panel.csv/parquet)")
        else:
            try:
                df = pd.read_csv(master_file) if master_file.suffix == '.csv' else pd.read_parquet(master_file)

                if len(df) < 100:
                    errors.append(f"Master data terlalu sedikit ({len(df)} baris) - minimal 100 baris diperlukan")

                if 'periode' in df.columns:
                    df['periode'] = pd.to_datetime(df['periode'])
                    latest_period = df['periode'].max()
                    months_old = (
                        (datetime.now().year - latest_period.year) * 12
                        + (datetime.now().month - latest_period.month)
                    )
                    newest_data_period_str = latest_period.strftime('%Y-%m')
                    if months_old > 6:
                        warnings.append(
                            f"Data terakhir adalah {latest_period.strftime('%Y-%m')} ({months_old} bulan yang lalu). "
                            f"Disarankan upload data terbaru untuk meningkatkan akurasi."
                        )
                    elif months_old > 3:
                        warnings.append(
                            f"Data terakhir adalah {latest_period.strftime('%Y-%m')} ({months_old} bulan yang lalu)."
                        )
            except Exception as e:
                errors.append(f"Error membaca master data: {str(e)}")

        # ── Uploads check (optional — informational only) ─────────────────────
        uploads_dir = data_dir / 'uploads'
        if uploads_dir.exists():
            data_files = [
                f for f in uploads_dir.iterdir()
                if f.suffix.lower() in ('.csv', '.xlsx', '.parquet', '.xls')
            ]
            if data_files:
                newest_file = max(data_files, key=lambda f: f.stat().st_mtime)
                try:
                    if newest_file.suffix == '.csv':
                        df_upload = pd.read_csv(newest_file)
                    elif newest_file.suffix in ('.xlsx', '.xls'):
                        df_upload = pd.read_excel(newest_file)
                    else:
                        df_upload = pd.read_parquet(newest_file)

                    if 'periode' in df_upload.columns:
                        df_upload['periode'] = pd.to_datetime(df_upload['periode'])
                        upload_min = df_upload['periode'].min()
                        upload_max = df_upload['periode'].max()

                        # Validate continuity against master
                        if master_file_exists:
                            try:
                                df_m = pd.read_csv(master_file) if master_file.suffix == '.csv' else pd.read_parquet(master_file)
                                df_m['periode'] = pd.to_datetime(df_m['periode'])
                                master_max = df_m['periode'].max()
                                # gap = months between last existing data and first new data
                                # gap = 1 means continuous (next month), gap = 0 means overlap
                                gap = (upload_min.year - master_max.year) * 12 + (upload_min.month - master_max.month)
                                
                                if gap > 2:
                                    errors.append(
                                        f"Data tidak kontinyu! Ada gap {gap-1} bulan antara data yang ada "
                                        f"({master_max.strftime('%Y-%m')}) dan data baru ({upload_min.strftime('%Y-%m')}). "
                                        f"Silakan lengkapi data yang hilang."
                                    )
                                elif gap == 2:
                                    warnings.append(
                                        f"Ada gap 1 bulan antara data yang ada ({master_max.strftime('%Y-%m')}) "
                                        f"dan data baru ({upload_min.strftime('%Y-%m')}). "
                                        f"Disarankan melengkapi data bulan yang hilang."
                                    )
                                elif gap < 1:
                                    # gap = 0 or negative means overlap
                                    if upload_max <= master_max:
                                        warnings.append(
                                            f"Data baru ({upload_min.strftime('%Y-%m')} s/d {upload_max.strftime('%Y-%m')}) "
                                            f"sepenuhnya overlap dengan data yang ada. Data lama akan ditimpa."
                                        )
                                    else:
                                        warnings.append(
                                            f"Data baru ({upload_min.strftime('%Y-%m')} s/d {upload_max.strftime('%Y-%m')}) "
                                            f"sebagian overlap dengan data yang ada ({master_max.strftime('%Y-%m')}). "
                                            f"Data yang overlap akan ditimpa."
                                        )
                                # gap == 1 is perfect continuity, no warning needed
                            except Exception:
                                pass

                        has_new_upload = True
                        newest_file_str = str(newest_file)
                        # Override period display with upload range
                        newest_data_period_str = f"{upload_min.strftime('%Y-%m')} s/d {upload_max.strftime('%Y-%m')} (upload baru)"
                except Exception as e:
                    warnings.append(f"Tidak dapat membaca file upload: {str(e)}")

        # Uploads are optional: if absent, retraining uses existing master data
        if not has_new_upload:
            warnings.append("Belum ada data baru di staging. Retraining akan menggunakan data master yang sudah ada.")

        is_ready = len(errors) == 0 and master_file_exists

        return is_ready, {
            'is_ready': is_ready,
            'errors': errors,
            'warnings': warnings,
            'checks': {
                'master_data_exists': master_file_exists,
                'has_new_upload': has_new_upload,
            },
            'newest_file': newest_file_str,
            'newest_data_period': newest_data_period_str,
        }
