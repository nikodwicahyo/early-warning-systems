"""Tests for src/rbbr_scoring.py"""

import pytest
import numpy as np
import pandas as pd

from src.rbbr_scoring import (
    score_npl, score_car, score_ldr, score_roa, score_bopo,
    compute_rbbr_score, score_to_pk, pk_to_sinyal, batch_score_rbbr,
)


class TestScoreNpl:
    def test_pk1(self):   assert score_npl(0.0) == 1
    def test_pk1_upper(self): assert score_npl(1.9) == 1
    def test_pk2(self):   assert score_npl(2.0) == 2
    def test_pk3(self):   assert score_npl(5.0) == 3
    def test_pk4(self):   assert score_npl(8.0) == 4
    def test_pk5(self):   assert score_npl(12.0) == 5
    def test_nan(self):   assert score_npl(np.nan) == 3


class TestScoreCar:
    def test_pk1(self):   assert score_car(20.0) == 1
    def test_pk2(self):   assert score_car(13.0) == 2
    def test_pk3(self):   assert score_car(11.0) == 3
    def test_pk4(self):   assert score_car(9.0) == 4
    def test_pk5(self):   assert score_car(5.0) == 5
    def test_nan(self):   assert score_car(np.nan) == 3


class TestScoreLdr:
    def test_pk1(self):   assert score_ldr(85.0) == 1
    def test_pk2_low(self): assert score_ldr(70.0) == 2
    def test_pk2_high(self): assert score_ldr(95.0) == 2
    def test_nan(self):   assert score_ldr(np.nan) == 3
    def test_returns_int(self): assert isinstance(score_ldr(80.0), int)


class TestScoreRoa:
    def test_pk1(self):   assert score_roa(2.0) == 1
    def test_pk2(self):   assert score_roa(1.3) == 2
    def test_pk3(self):   assert score_roa(1.0) == 3
    def test_pk4(self):   assert score_roa(0.3) == 4
    def test_pk5(self):   assert score_roa(-1.0) == 5
    def test_nan(self):   assert score_roa(np.nan) == 3


class TestScoreBopo:
    def test_pk1(self):   assert score_bopo(65.0) == 1
    def test_pk2(self):   assert score_bopo(75.0) == 2
    def test_pk3(self):   assert score_bopo(82.0) == 3
    def test_pk4(self):   assert score_bopo(87.0) == 4
    def test_pk5(self):   assert score_bopo(95.0) == 5
    def test_nan(self):   assert score_bopo(np.nan) == 3


class TestComputeRbbrScore:
    def _row(self, car=15, npl=1, ldr=85, roa=2, bopo=65):
        return {'CAR': car, 'NPL_gross': npl, 'LDR': ldr, 'ROA': roa, 'BOPO': bopo}

    def test_returns_float(self):
        assert isinstance(compute_rbbr_score(self._row()), float)

    def test_range(self):
        score = compute_rbbr_score(self._row())
        assert 1.0 <= score <= 5.0

    def test_healthy_bank_low_score(self):
        score = compute_rbbr_score(self._row(car=20, npl=0.5, ldr=85, roa=2.5, bopo=60))
        assert score < 3.0

    def test_unhealthy_bank_high_score(self):
        score = compute_rbbr_score(self._row(car=5, npl=15, ldr=150, roa=-2, bopo=100))
        assert score > 3.0

    def test_missing_keys_uses_default(self):
        # Should not raise; missing keys default to 3
        score = compute_rbbr_score({})
        assert isinstance(score, float)


class TestScoreToPk:
    def test_pk1(self):  assert score_to_pk(1.0) == 1
    def test_pk2(self):  assert score_to_pk(2.0) == 2
    def test_pk3(self):  assert score_to_pk(3.0) == 3
    def test_pk4(self):  assert score_to_pk(4.0) == 4
    def test_pk5(self):  assert score_to_pk(5.0) == 5
    def test_boundary_1_5(self): assert score_to_pk(1.5) == 2  # < 2.5 → pk2
    def test_boundary_2_5(self): assert score_to_pk(2.5) == 3


class TestPkToSinyal:
    def test_pk1_hijau(self):  assert pk_to_sinyal(1) == 'hijau'
    def test_pk2_hijau(self):  assert pk_to_sinyal(2) == 'hijau'
    def test_pk3_kuning(self): assert pk_to_sinyal(3) == 'kuning'
    def test_pk4_merah(self):  assert pk_to_sinyal(4) == 'merah'
    def test_pk5_merah(self):  assert pk_to_sinyal(5) == 'merah'


class TestBatchScoreRbbr:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame([
            {'CAR': 15.0, 'NPL_gross': 1.0, 'LDR': 85.0, 'ROA': 2.0, 'BOPO': 65.0},
            {'CAR': 8.0,  'NPL_gross': 10.0, 'LDR': 130.0, 'ROA': -1.0, 'BOPO': 95.0},
            {'CAR': 12.0, 'NPL_gross': 3.0, 'LDR': 90.0, 'ROA': 1.0, 'BOPO': 78.0},
        ])

    def test_output_columns(self, sample_df):
        result = batch_score_rbbr(sample_df)
        for col in ('rbbr_score', 'pk_computed', 'sinyal_ews'):
            assert col in result.columns

    def test_row_count_preserved(self, sample_df):
        result = batch_score_rbbr(sample_df)
        assert len(result) == len(sample_df)

    def test_pk_range(self, sample_df):
        result = batch_score_rbbr(sample_df)
        assert result['pk_computed'].between(1, 5).all()

    def test_sinyal_values(self, sample_df):
        result = batch_score_rbbr(sample_df)
        assert set(result['sinyal_ews']).issubset({'hijau', 'kuning', 'merah'})

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        batch_score_rbbr(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_pillar_score_columns(self, sample_df):
        result = batch_score_rbbr(sample_df)
        for col in ('score_npl', 'score_car', 'score_ldr', 'score_roa', 'score_bopo'):
            assert col in result.columns
