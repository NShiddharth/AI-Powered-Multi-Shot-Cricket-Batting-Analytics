"""
Test Suite for Biomechanical Metrics Statistical Analysis & Box Plot Visualizer
Verifies all requirements:
1. Treats each biomechanical metric independently.
2. Supports metrics: Elbow Angle, Head-Knee Distance, Foot Angle, Spine Lean,
   and any other numerical biomechanical metric present in the dataset.
3. Calculates statistical summary across all reference videos:
   Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR, P10, and P90.
4. Generates box plots conveying central tendency, spread, variability, and potential outliers.
5. Keeps empirical statistical distributions strictly separate from heuristic scoring thresholds.
6. Clearly labels statistical analysis as calculated from available reference/calibration data.
Compatible with both pytest and python -m unittest discover.
"""

import os
import json
import math
import tempfile
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import pytest
except ImportError:
    class _Approx:
        def __init__(self, expected, rel=0.01, abs=None):
            self.expected = expected
            self.rel = rel
            self.abs = abs
        def __eq__(self, actual):
            tol = self.abs if self.abs is not None else max(1e-6, abs(self.expected) * self.rel)
            return math.isclose(actual, self.expected, abs_tol=tol, rel_tol=self.rel)
        def __repr__(self):
            return f"approx({self.expected} ± {self.rel})"

    class _PytestMock:
        approx = _Approx
        @staticmethod
        def fixture(fn):
            return fn
    pytest = _PytestMock()

from src.calibration.biomechanical_statistics import (
    compute_metric_statistics,
    BiomechanicalStatisticsAnalyzer,
    get_metric_display_name,
    DATASET_LABEL
)


class TestBiomechanicalMetricIndependentStatistics(unittest.TestCase):
    """Tests that metrics are treated completely independently and all 11 descriptors are computed."""

    def test_compute_metric_statistics_descriptors(self):
        # Known synthetic sample
        sample_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats = compute_metric_statistics(
            values=sample_values,
            metric_key="front_elbow_angle",
            metric_name="Front Elbow Angle (°)"
        )

        assert stats['valid_samples_count'] == 10
        assert stats['mean'] == pytest.approx(55.0, 0.01)
        assert stats['median'] == pytest.approx(55.0, 0.01)
        assert stats['min'] == pytest.approx(10.0, 0.01)
        assert stats['max'] == pytest.approx(100.0, 0.01)
        assert stats['std_dev'] == pytest.approx(float(np.std(sample_values)), 0.01)
        assert stats['variance'] == pytest.approx(float(np.var(sample_values)), 0.01)

        # Percentiles
        expected_p10 = float(np.percentile(sample_values, 10))
        expected_q1 = float(np.percentile(sample_values, 25))
        expected_q3 = float(np.percentile(sample_values, 75))
        expected_p90 = float(np.percentile(sample_values, 90))
        expected_iqr = expected_q3 - expected_q1

        assert stats['p10'] == pytest.approx(expected_p10, 0.01)
        assert stats['q1'] == pytest.approx(expected_q1, 0.01)
        assert stats['q3'] == pytest.approx(expected_q3, 0.01)
        assert stats['iqr'] == pytest.approx(expected_iqr, 0.01)
        assert stats['p90'] == pytest.approx(expected_p90, 0.01)

        # Label check
        assert DATASET_LABEL in stats['data_source_label']

    def test_outlier_detection_via_tukey_iqr(self):
        # Values with clear outliers outside 1.5 * IQR
        # Normal range ~ [20, 25] with extreme 0 and 150
        sample_vals = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 0.0, 150.0]
        vids = [f"vid_{i}" for i in range(len(sample_vals))]
        stats = compute_metric_statistics(
            values=sample_vals,
            metric_key="spine_lean",
            video_identifiers=vids
        )

        outliers = stats['outliers']
        outlier_vals = [o['value'] for o in outliers]
        assert 150.0 in outlier_vals
        assert 0.0 in outlier_vals

    def test_independent_metric_treatment_with_missing_values(self):
        # Metric A has 10 samples, Metric B has 8 samples (with Nones)
        metric_a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        metric_b = [5.0, None, 15.0, None, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]

        stat_a = compute_metric_statistics(metric_a, "front_elbow_angle")
        stat_b = compute_metric_statistics(metric_b, "head_knee_alignment")

        assert stat_a['valid_samples_count'] == 10
        assert stat_b['valid_samples_count'] == 8
        # Ensure calculation of metric A is completely independent of metric B
        assert stat_a['mean'] == pytest.approx(55.0, 0.01)
        assert stat_b['mean'] == pytest.approx(40.0, 0.01)


class TestReferenceCalibrationDatasetAnalyzer(unittest.TestCase):
    """Tests analysis on actual reference calibration files."""

    @pytest.fixture
    def cover_drive_ref_path(self):
        return Path("output/reference_calibration/cover_drive/cover_drive_reference.json")

    def _resolve_ref_path(self, cover_drive_ref_path=None):
        if cover_drive_ref_path is not None and isinstance(cover_drive_ref_path, Path):
            return cover_drive_ref_path
        return Path("output/reference_calibration/cover_drive/cover_drive_reference.json")

    def test_cover_drive_reference_dataset_loading(self, cover_drive_ref_path=None):
        ref_path = self._resolve_ref_path(cover_drive_ref_path)
        assert ref_path.exists(), "Cover drive reference json must exist"
        analyzer = BiomechanicalStatisticsAnalyzer(ref_path, shot_type="cover_drive")

        # Must include all explicitly mentioned metrics:
        # Elbow Angle, Head-Knee Distance, Foot Angle, Spine Lean
        assert 'front_elbow_angle' in analyzer.metric_keys
        assert 'head_knee_alignment' in analyzer.metric_keys
        assert 'front_foot_direction' in analyzer.metric_keys
        assert 'spine_lean' in analyzer.metric_keys

        # Must also include any other numerical biomechanical metrics already present:
        for extra_key in ['shoulder_rotation', 'hip_rotation', 'stride_length', 'head_stability', 'follow_through_angle']:
            assert extra_key in analyzer.metric_keys

    def test_summary_dataframe_contains_all_11_descriptors(self, cover_drive_ref_path=None):
        ref_path = self._resolve_ref_path(cover_drive_ref_path)
        analyzer = BiomechanicalStatisticsAnalyzer(ref_path, shot_type="cover_drive")
        df = analyzer.get_summary_dataframe()

        required_columns = [
            'Biomechanical Metric', 'Valid Videos',
            'Mean', 'Median', 'Std Dev', 'Variance',
            'Min', 'Max', 'Q1', 'Q3', 'IQR', 'P10', 'P90'
        ]
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing from summary table"

        # Verify values are populated for elbow angle, spine lean, head-knee distance, foot angle
        elbow_row = df[df['Metric Key'] == 'front_elbow_angle'].iloc[0]
        assert elbow_row['Valid Videos'] == 10
        assert elbow_row['Mean'] > 0
        assert elbow_row['Median'] > 0
        assert elbow_row['Std Dev'] > 0
        assert elbow_row['Variance'] > 0
        assert elbow_row['Min'] > 0
        assert elbow_row['Max'] > elbow_row['Min']
        assert elbow_row['Q1'] > 0
        assert elbow_row['Q3'] >= elbow_row['Q1']
        assert elbow_row['IQR'] >= 0
        assert elbow_row['P10'] > 0
        assert elbow_row['P90'] > elbow_row['P10']

    def test_separation_from_hardcoded_scoring_thresholds(self, cover_drive_ref_path=None):
        """
        Verify that statistical values reflect empirical dataset distribution
        and are NOT hardcoded single threshold values.
        """
        ref_path = self._resolve_ref_path(cover_drive_ref_path)
        analyzer = BiomechanicalStatisticsAnalyzer(ref_path, shot_type="cover_drive")
        st = analyzer.statistics['front_elbow_angle']

        # Empirical mean and median should be actual floating point numbers calculated from 10 videos
        assert st['mean'] == pytest.approx(120.45, 0.5)
        assert st['median'] == pytest.approx(133.315, 0.5)
        # Not replaced by a single threshold like 90.0 or 70.0
        assert st['mean'] != 90.0
        assert st['median'] != 90.0

    def test_label_identifies_reference_calibration_data(self, cover_drive_ref_path=None):
        ref_path = self._resolve_ref_path(cover_drive_ref_path)
        analyzer = BiomechanicalStatisticsAnalyzer(ref_path, shot_type="cover_drive")
        for mk, st in analyzer.statistics.items():
            assert "reference/calibration data across all reference videos" in st['data_source_label']

    def test_box_plot_generation(self, cover_drive_ref_path=None, tmp_path=None):
        ref_path = self._resolve_ref_path(cover_drive_ref_path)
        analyzer = BiomechanicalStatisticsAnalyzer(ref_path, shot_type="cover_drive")

        if tmp_path is None:
            tmp_dir = Path(tempfile.mkdtemp())
        else:
            tmp_dir = Path(tmp_path)

        # Multi-panel box plots
        save_all = tmp_dir / "test_all_boxplots.png"
        fig_all = analyzer.plot_all_metric_boxplots(save_path=save_all)
        assert save_all.exists()
        assert save_all.stat().st_size > 1000

        # Single metric box plot (e.g. Elbow Angle)
        save_single = tmp_dir / "test_elbow_boxplot.png"
        fig_single = analyzer.plot_single_metric_boxplot("front_elbow_angle", save_path=save_single)
        assert save_single.exists()
        assert save_single.stat().st_size > 1000


if __name__ == "__main__":
    unittest.main()
