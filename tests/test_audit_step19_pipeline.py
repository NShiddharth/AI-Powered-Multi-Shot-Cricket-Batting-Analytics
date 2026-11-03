"""
Comprehensive Test Suite for Step 19 Verification
Explicitly verifies all 11 scenarios required by AthleteRise multi-shot specification:
TEST 1: Known cover-drive reference-like video -> Expected: cover_drive
TEST 2: Known straight-drive video -> Expected: straight_drive
TEST 3: Known pull-shot video -> Expected: pull_shot
TEST 4: Completely unrelated video -> Expected: UNKNOWN
TEST 5: Insufficient pose tracking -> Expected: UNKNOWN
TEST 6: Bat unavailable but ball available -> Expected contact: UNKNOWN
TEST 7: Ball unavailable near contact -> Expected contact: UNKNOWN
TEST 8: Bat and ball reliably tracked and clearly meet -> Expected: CONTACT
TEST 9: Bat and ball reliably tracked and clearly miss -> Expected: NO_CONTACT
TEST 10: Missing contact components -> Expected: UNKNOWN unless enough evidence remains
TEST 11: Existing cover-drive pipeline -> Works reliably without regression
"""

import unittest
import numpy as np
from src.shot_analysis.shot_config import SHOT_COVER_DRIVE, SHOT_STRAIGHT_DRIVE, SHOT_PULL_SHOT, SHOT_UNKNOWN
from src.shot_analysis.shot_classifier import ShotClassifier
from src.contact_detection.contact_estimator import ContactEstimator, STATUS_CONTACT, STATUS_NO_CONTACT, STATUS_UNKNOWN
from src.evaluation.scoring_engine import MultiShotScoringEngine

class TestStep19Pipeline(unittest.TestCase):

    def setUp(self):
        self.classifier = ShotClassifier(confidence_threshold=0.50, margin_threshold=0.12)
        self.contact_estimator = ContactEstimator(spatial_threshold_px=75.0, deflection_threshold_deg=25.0)

    # TEST 1: Known cover-drive reference-like video
    def test_01_known_cover_drive_prototype(self):
        features = dict(self.classifier.class_prototypes[SHOT_COVER_DRIVE]['means'])
        res = self.classifier.predict(features, debug=False)
        self.assertEqual(res['predicted_shot'], SHOT_COVER_DRIVE)
        self.assertGreaterEqual(res['confidence'], 0.50)
        self.assertEqual(res['classification_status'], 'classified')

    # TEST 2: Known straight-drive video
    def test_02_known_straight_drive_prototype(self):
        features = dict(self.classifier.class_prototypes[SHOT_STRAIGHT_DRIVE]['means'])
        res = self.classifier.predict(features, debug=False)
        self.assertEqual(res['predicted_shot'], SHOT_STRAIGHT_DRIVE)
        self.assertGreaterEqual(res['confidence'], 0.50)
        self.assertEqual(res['classification_status'], 'classified')

    # TEST 3: Known pull-shot video
    def test_03_known_pull_shot_prototype(self):
        features = dict(self.classifier.class_prototypes[SHOT_PULL_SHOT]['means'])
        res = self.classifier.predict(features, debug=False)
        self.assertEqual(res['predicted_shot'], SHOT_PULL_SHOT)
        self.assertGreaterEqual(res['confidence'], 0.50)
        self.assertEqual(res['classification_status'], 'classified')

    # TEST 4: Completely unrelated video
    def test_04_completely_unrelated_video(self):
        # Exaggerated values completely outside human cricket biomechanics
        unrelated_features = {
            'front_foot_direction': 10.0,
            'stride_length': 1.50,
            'shoulder_rotation': 10.0,
            'hip_rotation': 15.0,
            'front_elbow_angle': 5.0,
            'spine_lean': 75.0,
            'head_knee_alignment': 2.50,
            'rotary_shoulder_diff': 5.0,
            'wrist_horizontal_travel': 0.02,
            'wrist_vertical_travel': 0.02,
            'follow_through_angle': 10.0,
            'head_stability': 150.0
        }
        res = self.classifier.predict(unrelated_features, debug=False)
        self.assertEqual(res['predicted_shot'], SHOT_UNKNOWN)
        self.assertEqual(res['classification_status'], 'unknown')

    # TEST 5: Insufficient pose tracking
    def test_05_insufficient_pose_tracking(self):
        # 4 required features missing
        missing_features = {
            'front_foot_direction': None,
            'stride_length': None,
            'shoulder_rotation': None,
            'hip_rotation': None,
            'front_elbow_angle': 85.0,
            'spine_lean': 15.0,
            'head_knee_alignment': 0.25,
            'rotary_shoulder_diff': 60.0,
            'wrist_horizontal_travel': 0.40,
            'wrist_vertical_travel': 0.60,
            'follow_through_angle': 110.0,
            'head_stability': 20.0
        }
        res = self.classifier.predict(missing_features, debug=False)
        self.assertEqual(res['predicted_shot'], SHOT_UNKNOWN)
        self.assertEqual(res['classification_status'], 'unknown')
        self.assertIn("missing", res['reason'].lower())

    # TEST 6: Bat unavailable but ball available
    def test_06_bat_unavailable_ball_available(self):
        pose_impact = 10
        total_frames = 20
        bat_detections = {} # Bat completely untracked
        ball_detections = {
            8: {'center': [120, 60], 'radius': 6.0},
            9: {'center': [120, 90], 'radius': 6.0},
            10: {'center': [120, 120], 'radius': 6.0},
            11: {'center': [120, 150], 'radius': 6.0}
        }
        res = self.contact_estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_UNKNOWN)
        self.assertFalse(res['bat_tracking_available'])
        self.assertIsNone(res['visual_contact_frame'])

    # TEST 7: Ball unavailable near contact
    def test_07_ball_unavailable_near_contact(self):
        pose_impact = 10
        total_frames = 20
        bat_detections = {
            10: {'bbox': [100, 100, 40, 100], 'segment': [[120, 100], [120, 200]], 'confidence': 0.85}
        }
        ball_detections = {} # Ball completely untracked
        res = self.contact_estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_UNKNOWN)
        self.assertFalse(res['ball_tracking_available'])
        self.assertIsNone(res['visual_contact_frame'])
        self.assertIn("unavailable", res['reason'].lower())

    # TEST 8: Bat and ball reliably tracked and clearly meet
    def test_08_bat_and_ball_clearly_meet(self):
        pose_impact = 10
        total_frames = 20
        bat_detections = {
            10: {'bbox': [100, 100, 40, 100], 'segment': [[120, 100], [120, 200]], 'confidence': 0.90}
        }
        ball_detections = {
            8: {'center': [120, 60], 'radius': 6.0},
            9: {'center': [120, 90], 'radius': 6.0},
            10: {'center': [122, 120], 'radius': 6.0}, # 2px distance
            11: {'center': [160, 150], 'radius': 6.0}, # strong 45 deg deflection
            12: {'center': [195, 180], 'radius': 6.0}
        }
        res = self.contact_estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_CONTACT)
        self.assertEqual(res['visual_contact_frame'], 10)
        self.assertGreaterEqual(res['confidence'], 75.0)

    # TEST 9: Bat and ball reliably tracked and clearly miss
    def test_09_bat_and_ball_clearly_miss(self):
        pose_impact = 10
        total_frames = 20
        bat_detections = {
            10: {'bbox': [100, 100, 40, 100], 'segment': [[120, 100], [120, 200]], 'confidence': 0.85}
        }
        ball_detections = {
            8: {'center': [350, 60], 'radius': 6.0},
            9: {'center': [350, 90], 'radius': 6.0},
            10: {'center': [350, 120], 'radius': 6.0}, # 230px distance
            11: {'center': [350, 150], 'radius': 6.0}, # straight path past bat
            12: {'center': [350, 180], 'radius': 6.0}
        }
        res = self.contact_estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_NO_CONTACT)
        self.assertIsNone(res['visual_contact_frame'])
        self.assertGreaterEqual(res['confidence'], 85.0)

    # TEST 10: Missing contact components (only random noisy flickers far away)
    def test_10_missing_contact_components_returns_unknown(self):
        pose_impact = 10
        total_frames = 20
        bat_detections = {
            10: {'bbox': [100, 100, 40, 100], 'segment': [[120, 100], [120, 200]], 'confidence': 0.80}
        }
        # Incoherent detections that jump across the screen (noise)
        ball_detections = {
            8: {'center': [50, 50], 'radius': 5.0},
            12: {'center': [500, 500], 'radius': 5.0} # 636px jump
        }
        res = self.contact_estimator.estimate_contact(pose_impact, bat_detections, ball_detections, total_frames)
        self.assertEqual(res['status'], STATUS_UNKNOWN)
        self.assertIsNone(res['visual_contact_frame'])

    # TEST 11: Existing cover-drive scoring pipeline functionality preserved
    def test_11_existing_cover_drive_pipeline(self):
        engine = MultiShotScoringEngine(shot_type=SHOT_COVER_DRIVE, config_path="config.json")
        sample_metrics = {
            'front_elbow_angle': 85.0,
            'spine_lean': 17.0,
            'head_knee_alignment': 0.22,
            'front_foot_direction': 160.0,
            'shoulder_rotation': 135.0,
            'hip_rotation': 138.0,
            'stride_length': 0.26,
            'head_stability': 20.0,
            'follow_through_angle': 105.0,
            'follow_through_quality': 70.0
        }
        eval_res = engine.evaluate(user_metrics=sample_metrics, handedness="right")
        self.assertEqual(eval_res['shot_type'], SHOT_COVER_DRIVE)
        self.assertIn('overall_score', eval_res)
        self.assertGreater(eval_res['overall_score'], 60.0)
        self.assertEqual(len(eval_res['category_scores']), 5)
        self.assertGreaterEqual(len(eval_res['metric_scores']), 10)
