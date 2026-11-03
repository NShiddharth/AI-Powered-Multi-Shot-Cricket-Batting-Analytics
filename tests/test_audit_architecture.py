"""
Comprehensive Verification Test Suite for Forensic Audit Architecture
Tests continuous scoring policies, 5-category hierarchical weighting,
dynamic N/A renormalization, follow-through bug fix, stance handling, and fallback behavior.
"""

import os
import sys
import json
import unittest
import numpy as np
from pathlib import Path

# Ensure repository root is in sys.path when running tests directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.scoring_engine import (
    CoverDriveScoringEngine,
    score_metric_continuous,
    map_score_to_grade,
    METRIC_POLICIES,
    CATEGORY_METRIC_MAPPING,
    OVERALL_CATEGORY_WEIGHTS
)
import cover_drive_analysis_realtime

class TestScoringEngine(unittest.TestCase):

    def setUp(self):
        self.engine = CoverDriveScoringEngine(
            reference_path="output/reference_calibration/cover_drive/cover_drive_reference.json",
            config_path="config.json"
        )
        self.assertTrue(self.engine.is_reference_valid())
        self.ref_stats = self.engine.reference_data['statistics']

    def test_continuous_scoring_policies(self):
        # 1. Two-Sided Optimal (e.g. front_foot_direction)
        stat = self.ref_stats['front_foot_direction']
        # Median should get 95-100
        sc_med, _ = score_metric_continuous('front_foot_direction', stat['median'], stat)
        self.assertGreaterEqual(sc_med, 95.0)

        # Within IQR [P25, P75]
        sc_iqr, _ = score_metric_continuous('front_foot_direction', (stat['p25'] + stat['p75']) / 2.0, stat)
        self.assertGreaterEqual(sc_iqr, 90.0)

        # In acceptable range [P10, P25]
        sc_acc, _ = score_metric_continuous('front_foot_direction', stat['p10'], stat)
        self.assertGreaterEqual(sc_acc, 70.0)
        self.assertLess(sc_acc, 90.0)

        # Severe Outlier
        sc_out, _ = score_metric_continuous('front_foot_direction', 20.0, stat)
        self.assertLess(sc_out, 50.0)

        # 2. Lower is Better (e.g. head_knee_alignment, spine_lean, head_stability)
        stat_hk = self.ref_stats['head_knee_alignment']
        sc_hk_low, _ = score_metric_continuous('head_knee_alignment', stat_hk['p25'] * 0.8, stat_hk)
        self.assertGreaterEqual(sc_hk_low, 95.0)

        sc_hk_high, _ = score_metric_continuous('head_knee_alignment', stat_hk['p90'] * 1.5, stat_hk)
        self.assertLess(sc_hk_high, 60.0)

        # 3. Higher is Better (e.g. front_elbow_angle)
        stat_el = self.ref_stats['front_elbow_angle']
        sc_el_high, _ = score_metric_continuous('front_elbow_angle', stat_el['p90'], stat_el)
        self.assertGreaterEqual(sc_el_high, 95.0)

        sc_el_low, _ = score_metric_continuous('front_elbow_angle', stat_el['p10'] * 0.5, stat_el)
        self.assertLess(sc_el_low, 60.0)

        # 4. N/A handling
        sc_na, txt_na = score_metric_continuous('front_elbow_angle', None, stat_el)
        self.assertIsNone(sc_na)
        self.assertIn("N/A", txt_na)

    def test_dynamic_na_renormalization_within_category(self):
        # In Swing Control: front_elbow_angle (0.50), shoulder_rotation (0.25), hip_rotation (0.25)
        # If hip_rotation is None, front_elbow_angle should be 0.50 / 0.75 = 66.7% and shoulder_rotation 0.25 / 0.75 = 33.3%
        sample_metrics = {
            'front_elbow_angle': self.ref_stats['front_elbow_angle']['median'], # ~95-100 score
            'shoulder_rotation': self.ref_stats['shoulder_rotation']['median'], # ~95-100 score
            'hip_rotation': None, # N/A
            'front_foot_direction': self.ref_stats['front_foot_direction']['median'],
            'stride_length': self.ref_stats['stride_length']['median'],
            'head_knee_alignment': self.ref_stats['head_knee_alignment']['median'],
            'head_stability': self.ref_stats['head_stability']['median'],
            'spine_lean': self.ref_stats['spine_lean']['median'],
            'follow_through_angle': self.ref_stats['follow_through_angle']['median'],
            'follow_through_quality': self.ref_stats['follow_through_quality']['median']
        }

        res = self.engine.evaluate(sample_metrics, handedness='right')
        sc_cat = res['category_scores']['swing_control']
        self.assertEqual(sc_cat['status'], 'valid')
        self.assertIn('hip_rotation', sc_cat['excluded_metrics'])
        self.assertGreaterEqual(sc_cat['score'], 90.0)

    def test_dynamic_na_renormalization_entire_category(self):
        # If follow_through_angle and follow_through_quality are both None:
        sample_metrics = {
            'front_elbow_angle': self.ref_stats['front_elbow_angle']['median'],
            'shoulder_rotation': self.ref_stats['shoulder_rotation']['median'],
            'hip_rotation': self.ref_stats['hip_rotation']['median'],
            'front_foot_direction': self.ref_stats['front_foot_direction']['median'],
            'stride_length': self.ref_stats['stride_length']['median'],
            'head_knee_alignment': self.ref_stats['head_knee_alignment']['median'],
            'head_stability': self.ref_stats['head_stability']['median'],
            'spine_lean': self.ref_stats['spine_lean']['median'],
            'follow_through_angle': None,
            'follow_through_quality': None
        }

        res = self.engine.evaluate(sample_metrics, handedness='right')
        ft_cat = res['category_scores']['follow_through']
        self.assertIsNone(ft_cat['score'])
        self.assertEqual(ft_cat['status'], 'N/A')

        # Overall score should renormalize across remaining 4 categories (total weight 0.85)
        # Never convert N/A to zero!
        self.assertGreaterEqual(res['overall_score'], 90.0)
        self.assertEqual(res['skill_grade'], "Advanced / Elite Execution")

    def test_project_grade_mapping(self):
        self.assertEqual(map_score_to_grade(95.0), "Advanced / Elite Execution")
        self.assertEqual(map_score_to_grade(82.5), "Above Average / Proficient")
        self.assertEqual(map_score_to_grade(68.0), "Intermediate")
        self.assertEqual(map_score_to_grade(52.0), "Developing")
        self.assertEqual(map_score_to_grade(35.0), "Beginner")
        self.assertEqual(map_score_to_grade(None), "N/A")

    def test_fallback_expert_mode(self):
        # Simulate missing reference
        fallback_engine = CoverDriveScoringEngine(
            reference_path="non_existent_file.json",
            config_path="config.json"
        )
        self.assertFalse(fallback_engine.is_reference_valid())
        res = fallback_engine.evaluate({'front_elbow_angle': 130.0}, handedness='right')
        self.assertEqual(res['evaluation_mode'], 'FALLBACK EXPERT RULES')
        self.assertTrue(res['is_fallback'])
        self.assertIsNotNone(res['overall_score'])

    def test_config_json_zero_influence_on_primary_score(self):
        # Even if config.json thresholds are extreme or different, primary score must be identical
        engine1 = CoverDriveScoringEngine(
            reference_path="output/reference_calibration/cover_drive/cover_drive_reference.json",
            config_path="config.json"
        )
        engine2 = CoverDriveScoringEngine(
            reference_path="output/reference_calibration/cover_drive/cover_drive_reference.json",
            config_path="non_existent_config.json"
        )
        sample_metrics = {
            'front_elbow_angle': 90.0,
            'shoulder_rotation': 140.0,
            'hip_rotation': 150.0,
            'front_foot_direction': 160.0,
            'stride_length': 0.20,
            'head_knee_alignment': 0.25,
            'head_stability': 20.0,
            'spine_lean': 8.0,
            'follow_through_angle': 115.0,
            'follow_through_quality': 75.0
        }
        res1 = engine1.evaluate(sample_metrics, handedness='right')
        res2 = engine2.evaluate(sample_metrics, handedness='right')
        self.assertEqual(res1['evaluation_mode'], 'DATA-DRIVEN REFERENCE')
        self.assertEqual(res2['evaluation_mode'], 'DATA-DRIVEN REFERENCE')
        self.assertEqual(res1['overall_score'], res2['overall_score'])
        self.assertEqual(res1['overall_score'], 83.97)

    def test_automatic_fallback_when_reference_missing(self):
        # Verify fallback works completely without requiring uploaded config
        engine_fallback = CoverDriveScoringEngine(
            reference_path="invalid_path_to_ref.json",
            config_path="config.json"
        )
        self.assertFalse(engine_fallback.is_reference_valid())
        res = engine_fallback.evaluate({
            'front_elbow_angle': 130.0,
            'head_knee_alignment': 0.2,
            'front_foot_direction': 90.0,
            'spine_lean': 15.0,
            'follow_through_angle': 130.0
        }, handedness='right')
        self.assertEqual(res['evaluation_mode'], 'FALLBACK EXPERT RULES')
        self.assertTrue(res['is_fallback'])
        self.assertEqual(res['reference_sample_count'], 0)

if __name__ == '__main__':
    unittest.main()
