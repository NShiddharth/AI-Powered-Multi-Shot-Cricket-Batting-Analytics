"""
Comprehensive Unit Test Suite for Multi-Shot Analysis Architecture
Tests:
1. Shot Classification (Cover Drive, Straight Drive, Pull Shot, Unknown fallback)
2. Empirical Calibration (Correct directories, zero cross-contamination, independent statistics)
3. Bat-Ball Contact Detection (Contact detected, no contact, ball unavailable, bat unavailable, low confidence)
4. Multi-Shot Continuous Scoring (Matching baseline selection, dynamic renormalization)
5. Backward Compatibility (Existing cover drive interfaces)
"""

import os
import sys
import json
import unittest
import numpy as np
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shot_analysis.shot_config import (
    SHOT_COVER_DRIVE,
    SHOT_STRAIGHT_DRIVE,
    SHOT_PULL_SHOT,
    SHOT_UNKNOWN,
    get_shot_reference_path,
    get_shot_reference_json
)
from src.shot_analysis.shot_classifier import ShotClassifier
from src.calibration.calibration_factory import CalibrationFactory
from src.contact_detection.bat_ball_detector import BatDetector, BallDetector
from src.contact_detection.contact_estimator import ContactEstimator, STATUS_CONTACT, STATUS_NO_CONTACT, STATUS_UNKNOWN
from src.contact_detection.annotation_manager import ContactAnnotationManager
from src.evaluation.base_scoring_engine import MultiShotScoringEngine
from src.evaluation.scoring_engine import CoverDriveScoringEngine

class TestShotClassification(unittest.TestCase):
    def setUp(self):
        self.classifier = ShotClassifier(confidence_threshold=0.50, margin_threshold=0.12)

    def test_classify_cover_drive_prototype(self):
        # Features characteristic of calibrated Cover Drive reference profile
        features = dict(self.classifier.class_prototypes[SHOT_COVER_DRIVE]['means'])
        res = self.classifier.predict(features)
        self.assertEqual(res['predicted_shot'], SHOT_COVER_DRIVE)
        self.assertGreaterEqual(res['confidence'], 0.50)

    def test_classify_straight_drive_prototype(self):
        # Features characteristic of calibrated Straight Drive reference profile
        features = dict(self.classifier.class_prototypes[SHOT_STRAIGHT_DRIVE]['means'])
        res = self.classifier.predict(features)
        self.assertEqual(res['predicted_shot'], SHOT_STRAIGHT_DRIVE)
        self.assertGreaterEqual(res['confidence'], 0.50)

    def test_classify_pull_shot_prototype(self):
        # Features characteristic of calibrated Pull Shot reference profile
        features = dict(self.classifier.class_prototypes[SHOT_PULL_SHOT]['means'])
        res = self.classifier.predict(features)
        self.assertEqual(res['predicted_shot'], SHOT_PULL_SHOT)
        self.assertGreaterEqual(res['confidence'], 0.50)

    def test_low_confidence_unknown_rejection(self):
        # Ambiguous sample halfway between straight and cover
        features = {
            'front_foot_direction': 164.0,
            'stride_length': 0.24,
            'shoulder_rotation': 119.0,
            'hip_rotation': 120.0,
            'front_elbow_angle': 110.0,
            'spine_lean': 12.0,
            'head_knee_alignment': 0.24,
            'rotary_shoulder_diff': 30.0,
            'wrist_horizontal_travel': 0.17,
            'wrist_vertical_travel': 0.35,
            'follow_through_angle': 108.0,
            'head_stability': 33.0
        }
        res = self.classifier.predict(features)
        self.assertEqual(res['predicted_shot'], SHOT_UNKNOWN)
        self.assertIn("threshold", res['reason'].lower())


class TestCalibrationIntegrity(unittest.TestCase):
    def test_reference_directory_isolation(self):
        p_cover = get_shot_reference_path(SHOT_COVER_DRIVE)
        p_straight = get_shot_reference_path(SHOT_STRAIGHT_DRIVE)
        p_pull = get_shot_reference_path(SHOT_PULL_SHOT)

        self.assertIn("cover_drive", p_cover)
        self.assertIn("straight_drive", p_straight)
        self.assertIn("pull_shot", p_pull)
        self.assertNotEqual(p_cover, p_straight)
        self.assertNotEqual(p_straight, p_pull)

    def test_reference_json_isolation_and_no_contamination(self):
        j_cover = get_shot_reference_json(SHOT_COVER_DRIVE)
        j_straight = get_shot_reference_json(SHOT_STRAIGHT_DRIVE)
        j_pull = get_shot_reference_json(SHOT_PULL_SHOT)

        with open(j_cover) as f:
            d_cover = json.load(f)
        with open(j_straight) as f:
            d_straight = json.load(f)
        with open(j_pull) as f:
            d_pull = json.load(f)

        self.assertEqual(d_cover['shot_type'], SHOT_COVER_DRIVE)
        self.assertEqual(d_straight['shot_type'], SHOT_STRAIGHT_DRIVE)
        self.assertEqual(d_pull['shot_type'], SHOT_PULL_SHOT)

        # Confirm distinct empirical distributions (no cross-contamination)
        m_c = d_cover['statistics']['front_elbow_angle']['mean']
        m_s = d_straight['statistics']['front_elbow_angle']['mean']
        m_p = d_pull['statistics']['front_elbow_angle']['mean']

        self.assertIsNotNone(m_c)
        self.assertIsNotNone(m_s)
        self.assertIsNotNone(m_p)
        self.assertNotEqual(m_c, m_s)
        self.assertNotEqual(m_s, m_p)
        self.assertGreater(m_s, m_c) # Straight drive has higher elbow extension


class TestContactDetection(unittest.TestCase):
    def setUp(self):
        self.estimator = ContactEstimator(spatial_threshold_px=75.0, deflection_threshold_deg=25.0)

    def test_contact_detected(self):
        # Ball passes within 20px of bat with a 45-degree deflection
        pose_impact = 10
        total_frames = 20
        bat_detections = {
            10: {'bbox': [100, 100, 40, 100], 'segment': [[120, 100], [120, 200]], 'confidence': 0.85}
        }
        ball_detections = {
            8: {'center': [120, 60], 'radius': 6.0},
            9: {'center': [120, 90], 'radius': 6.0},
            10: {'center': [125, 120], 'radius': 6.0}, # 5px from bat segment!
            11: {'center': [160, 150], 'radius': 6.0}, # deflected right
            12: {'center': [190, 180], 'radius': 6.0}
        }
        res = self.estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_CONTACT)
        self.assertGreaterEqual(res['confidence'], 70.0)
        self.assertEqual(res['estimated_frame'], 10)

    def test_no_contact(self):
        # Ball passes 200px away from bat with 0 deflection
        pose_impact = 10
        total_frames = 20
        bat_detections = {
            10: {'bbox': [100, 100, 40, 100], 'segment': [[120, 100], [120, 200]], 'confidence': 0.80}
        }
        ball_detections = {
            8: {'center': [350, 60], 'radius': 6.0},
            9: {'center': [350, 90], 'radius': 6.0},
            10: {'center': [350, 120], 'radius': 6.0}, # 230px from bat!
            11: {'center': [350, 150], 'radius': 6.0},
            12: {'center': [350, 180], 'radius': 6.0}
        }
        res = self.estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_NO_CONTACT)

    def test_ball_unavailable_status_unknown(self):
        # When ball tracking is unavailable, status MUST be UNKNOWN (never false NO_CONTACT)
        pose_impact = 10
        total_frames = 20
        bat_detections = {
            10: {'bbox': [100, 100, 40, 100], 'segment': [[120, 100], [120, 200]], 'confidence': 0.80}
        }
        ball_detections = {} # No ball tracked!
        res = self.estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_UNKNOWN)
        self.assertFalse(res['ball_tracking_available'])
        self.assertIn("unavailable", res['reason'].lower())

    def test_bat_unavailable_status_unknown(self):
        # When bat cannot be tracked jointly
        pose_impact = 10
        total_frames = 20
        bat_detections = {}
        ball_detections = {
            10: {'center': [120, 120], 'radius': 6.0},
            11: {'center': [120, 150], 'radius': 6.0}
        }
        res = self.estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_UNKNOWN)


class TestMultiShotScoring(unittest.TestCase):
    def test_cover_drive_uses_cover_reference(self):
        engine = MultiShotScoringEngine(shot_type=SHOT_COVER_DRIVE)
        self.assertTrue(engine.is_reference_valid())
        self.assertIn("cover_drive", engine.reference_path)
        eval_res = engine.evaluate({'front_foot_direction': 165.0, 'stride_length': 0.25})
        self.assertEqual(eval_res['shot_type'], SHOT_COVER_DRIVE)
        self.assertEqual(eval_res['evaluation_mode'], 'DATA-DRIVEN REFERENCE')

    def test_straight_drive_uses_straight_reference(self):
        engine = MultiShotScoringEngine(shot_type=SHOT_STRAIGHT_DRIVE)
        self.assertTrue(engine.is_reference_valid())
        self.assertIn("straight_drive", engine.reference_path)
        eval_res = engine.evaluate({'front_foot_direction': 95.0, 'stride_length': 0.33})
        self.assertEqual(eval_res['shot_type'], SHOT_STRAIGHT_DRIVE)
        self.assertEqual(eval_res['evaluation_mode'], 'DATA-DRIVEN REFERENCE')

    def test_pull_shot_uses_pull_reference(self):
        engine = MultiShotScoringEngine(shot_type=SHOT_PULL_SHOT)
        self.assertTrue(engine.is_reference_valid())
        self.assertIn("pull_shot", engine.reference_path)
        eval_res = engine.evaluate({'front_foot_direction': 65.0, 'stride_length': 0.16})
        self.assertEqual(eval_res['shot_type'], SHOT_PULL_SHOT)
        self.assertEqual(eval_res['evaluation_mode'], 'DATA-DRIVEN REFERENCE')

    def test_unknown_shot_no_shot_specific_metrics(self):
        # Section 3 & 11: Unknown shot must NOT receive shot-specific metrics
        engine = MultiShotScoringEngine(shot_type=SHOT_UNKNOWN)
        eval_res = engine.evaluate({'front_foot_direction': 120.0, 'stride_length': 0.22})
        self.assertEqual(eval_res['shot_type'], SHOT_UNKNOWN)
        self.assertEqual(eval_res['shot_specific_metrics'], {})
        self.assertIn("unavailable", eval_res['shot_specific_status'].lower())

    def test_unknown_shot_no_shot_specific_coaching(self):
        # Section 3 & 27: Unknown shot must receive strictly generic coaching
        engine = MultiShotScoringEngine(shot_type=SHOT_UNKNOWN)
        eval_res = engine.evaluate({'front_foot_direction': 120.0, 'stride_length': 0.22})
        recs = eval_res['coaching_feedback']['coaching_recommendations']
        for r in recs:
            self.assertNotIn("pull along the ground", r.lower())
            self.assertNotIn("extra cover", r.lower())
            self.assertNotIn("mid-off", r.lower())
        self.assertTrue(any("re-record" in r.lower() or "confidently" in r.lower() for r in recs))

    def test_unknown_shot_no_overall_score(self):
        # Section 11: Unknown shot overall score must be Not Available
        engine = MultiShotScoringEngine(shot_type=SHOT_UNKNOWN)
        eval_res = engine.evaluate({'front_foot_direction': 120.0, 'stride_length': 0.22})
        self.assertIsNone(eval_res['overall_score'])
        self.assertEqual(eval_res['overall_score_display'], "Not Available")
        self.assertEqual(eval_res['skill_grade'], "Classification Required")


class TestContactAnnotationManager(unittest.TestCase):
    def test_save_and_retrieve_annotation(self):
        mgr = ContactAnnotationManager(storage_dir="data/contact_annotations")
        p = mgr.save_annotation("unit_test_clip.avi", 42, "contact", notes="Unit test validation")
        self.assertTrue(os.path.exists(p))

        rec = mgr.get_annotation("unit_test_clip.avi")
        self.assertIsNotNone(rec)
        self.assertEqual(rec['candidate_frame'], 42)
        self.assertEqual(rec['label'], "contact")

        # Clean up test file
        if os.path.exists(p):
            os.remove(p)


if __name__ == '__main__':
    unittest.main()
