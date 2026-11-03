"""
AthleteRise - Multi-Shot Real-Time & Offline Batting Analysis
Integrates:
- MediaPipe Pose Estimation & Handedness Detection
- Data-Driven Multi-Shot Classification (Cover Drive, Straight Drive, Pull Shot, Unknown)
- Shot-Specific Phase Segmentation & Biomechanical Impact Estimation
- Modular Bat & Ball Tracking & Visual Bat-Ball Contact Estimation
- Continuous Statistical Evaluation against Calibrated Shot Reference Databases
- Dynamic Occlusion Renormalization & Safe Expert Rule Fallback
- Rich Visual Video Annotation with Shot & Contact Overlays
"""

import cv2
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import os
from pathlib import Path

# MediaPipe import handling with robust fallback
try:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except Exception:
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
    except Exception:
        try:
            from mediapipe.tasks.python.vision import PoseLandmark as _PL
            class _FallbackMPPose:
                PoseLandmark = _PL
                POSE_CONNECTIONS = []
            mp_pose = _FallbackMPPose
            mp_drawing = None
        except Exception:
            mp_pose = None
            mp_drawing = None

from src.preprocessing.video_preprocessing import VideoPreprocessor
from src.biomechanics.metrics import BiomechanicsCalculator, detect_handedness, calculate_angle
from src.phase_detection.phase_detector import CoverDrivePhaseDetector, PhaseDetectorFactory
from src.evaluation.scoring_engine import CoverDriveScoringEngine, MultiShotScoringEngine
from src.shot_analysis.shot_config import SHOT_COVER_DRIVE, SHOT_STRAIGHT_DRIVE, SHOT_PULL_SHOT, SHOT_UNKNOWN, SHOT_DISPLAY_NAMES
from src.shot_analysis.shot_features import ShotFeatureExtractor
from src.shot_analysis.shot_classifier import ShotClassifier
from src.contact_detection.bat_ball_detector import BatDetector, BallDetector
from src.contact_detection.contact_estimator import ContactEstimator, STATUS_CONTACT, STATUS_UNKNOWN
from src.reporting.multi_shot_report import MultiShotReportGenerator

def main(video_input=None, config_input=None, shot_type_override=None):
    """
    Main execution pipeline for Multi-Shot Cricket Batting Analysis.
    Automatically identifies shot class, evaluates bat-ball contact,
    selects matching calibrated reference database, and scores execution.
    """
    # 1. Config path handling
    config_path = config_input or 'config.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            config = {}
    else:
        config = {
            "elbow_angle_threshold": 120,
            "head_knee_distance_threshold": 50,
            "foot_angle_lower_threshold": 45,
            "foot_angle_upper_threshold": 135,
            "spine_lean_threshold": 30
        }

    # 2. Resolve Input and Output video paths
    if video_input:
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Input video file not found: {video_input}")
        video_path = video_input
    elif os.path.exists('temp.mp4'):
        video_path = 'temp.mp4'
    elif os.path.exists('input_video.avi'):
        video_path = 'input_video.avi'
    elif os.path.exists('input_video.mp4'):
        video_path = 'input_video.mp4'
    else:
        video_path = 'temp.mp4'

    output_path = 'output/annotated_video.mp4'
    os.makedirs('output', exist_ok=True)

    # 3. Ingest video using VideoPreprocessor
    preprocessor = VideoPreprocessor(
        video_path=video_path,
        normalize_brightness=config.get('normalize_brightness', False),
        save_frames_dir="frames"
    )

    frame_width = preprocessor.width
    frame_height = preprocessor.height
    fps = preprocessor.fps or 25
    total_frames = max(1, preprocessor.total_frames)

    from src.pose_estimation.posture_tracker import VisionPostureTracker
    posture_tracker = VisionPostureTracker(mp_pose)
    posture_tracker.profile_video(video_path)

    bat_detector = BatDetector()
    ball_detector = BallDetector()

    raw_landmarks_list = []
    frame_metrics_list = []
    bat_detections = {}
    ball_detections = {}
    cached_frames = []

    calculator_dummy = BiomechanicsCalculator(handedness='right', frame_width=frame_width, frame_height=frame_height)
    elbow_angles = []
    spine_leans = []

    evaluation_expert = {
        "footwork": [],
        "head_position": [],
        "swing_control": [],
        "balance": [],
        "follow_through": []
    }

    prev_frame = None

    # First Pass: Read frames, extract pose landmarks & visual bat/ball candidates
    for frame_idx, frame in preprocessor.read_frames():
        cached_frames.append(frame.copy())
        lm_dict = posture_tracker.process_frame(frame, frame_idx, total_frames)

        raw_landmarks_list.append(lm_dict)
        frame_metrics_list.append({'frame_idx': frame_idx, 'landmarks': lm_dict})

        # Track bat & ball
        wrist_px = (int(lm_dict['left_wrist'][0] * frame_width), int(lm_dict['left_wrist'][1] * frame_height)) if lm_dict else None
        bat_res = bat_detector.detect(frame, wrist_point=wrist_px, frame_idx=frame_idx)
        if bat_res:
            bat_detections[frame_idx] = bat_res

        ball_res = ball_detector.detect_candidate(frame, prev_frame=prev_frame, frame_idx=frame_idx)
        if ball_res:
            ball_detections[frame_idx] = ball_res

        prev_frame = frame.copy()

    preprocessor.release()
    posture_tracker.close()

    # Determine Handedness & Compute Frame Metrics
    handedness = detect_handedness(raw_landmarks_list)
    calculator = BiomechanicsCalculator(handedness=handedness, frame_width=frame_width, frame_height=frame_height)

    processed_frame_metrics = []
    for fm in frame_metrics_list:
        lm = fm['landmarks']
        if lm is not None:
            if '_direct_metrics' in lm:
                m = lm['_direct_metrics']
            else:
                m = calculator.compute_frame_metrics(lm)
        else:
            m = None
        processed_frame_metrics.append({'frame_idx': fm['frame_idx'], 'metrics': m, 'landmarks': lm})
        if m:
            if m.get('front_elbow_angle') is not None:
                elbow_angles.append(m['front_elbow_angle'])
            if m.get('spine_lean') is not None:
                spine_leans.append(m['spine_lean'])

    # 4. Preliminary stroke impact estimation for feature extraction (shot-independent)
    prelim_detector = CoverDrivePhaseDetector()
    temp_contact_frame = prelim_detector.detect_contact_frame(processed_frame_metrics, total_frames)

    # 5. Extract Kinematic Feature Vector & Classify Shot
    feature_extractor = ShotFeatureExtractor(handedness=handedness)
    kinematic_features = feature_extractor.extract_features(processed_frame_metrics, temp_contact_frame, total_frames)

    cfg_data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg_data = json.load(f)
        except Exception:
            cfg_data = {}
    clf_cfg = cfg_data.get('classifier', {})
    conf_thresh = float(clf_cfg.get('confidence_threshold', 0.50))
    marg_thresh = float(clf_cfg.get('margin_threshold', 0.12))

    classifier = ShotClassifier(confidence_threshold=conf_thresh, margin_threshold=marg_thresh)
    classification_result = classifier.predict(kinematic_features)

    # Save classification debug JSON (Requirement 18)
    debug_payload = {
        "reference_counts": {
            s: len(classifier.reference_samples.get(s, [])) for s in [SHOT_COVER_DRIVE, SHOT_STRAIGHT_DRIVE, SHOT_PULL_SHOT]
        },
        "training_samples": sum(len(classifier.reference_samples.get(s, [])) for s in [SHOT_COVER_DRIVE, SHOT_STRAIGHT_DRIVE, SHOT_PULL_SHOT]),
        "feature_count": len(kinematic_features),
        "test_feature_count": len(kinematic_features),
        "extracted_features": kinematic_features,
        "missing_features": [k for k, v in kinematic_features.items() if v is None],
        "nan_features": [k for k, v in kinematic_features.items() if isinstance(v, float) and np.isnan(v)],
        "predicted_class": classification_result.get('predicted_shot'),
        "confidence": classification_result.get('confidence'),
        "class_probabilities": classification_result.get('class_probabilities', {}),
        "top1_top2_margin": classification_result.get('feature_evidence', {}).get('margin'),
        "classification_status": classification_result.get('classification_status'),
        "unknown_reason": classification_result.get('reason') if classification_result.get('predicted_shot') == SHOT_UNKNOWN else None
    }
    with open('output/classification_debug.json', 'w', encoding='utf-8') as df_out:
        json.dump(debug_payload, df_out, indent=4)

    # Resolve active shot type
    if shot_type_override and shot_type_override.lower() in [SHOT_COVER_DRIVE, SHOT_STRAIGHT_DRIVE, SHOT_PULL_SHOT]:
        active_shot = shot_type_override.lower()
    else:
        active_shot = classification_result['predicted_shot']

    # 6. Shot-Specific Phase & Biomechanical Impact Detection
    detector_shot = active_shot if active_shot != SHOT_UNKNOWN else SHOT_COVER_DRIVE
    phase_detector = PhaseDetectorFactory.create(detector_shot)
    pose_impact_frame = phase_detector.detect_contact_frame(processed_frame_metrics, total_frames)
    phases = phase_detector.segment_phases(processed_frame_metrics, pose_impact_frame, total_frames)

    # 7. Bat-Ball Contact Estimation
    contact_estimator = ContactEstimator(spatial_threshold_px=75.0, deflection_threshold_deg=25.0)
    contact_analysis = contact_estimator.estimate_contact(
        pose_impact_frame=pose_impact_frame,
        bat_detections=bat_detections,
        ball_detections=ball_detections,
        total_frames=total_frames
    )

    # 8. Representative Metrics at Contact Window
    c_min, c_max = phases.get('contact_window', [max(1, pose_impact_frame - 3), min(total_frames, pose_impact_frame + 3)])
    c_window_metrics = [
        fm['metrics'] for fm in processed_frame_metrics
        if fm['metrics'] is not None and c_min <= fm['frame_idx'] <= c_max
    ]
    if not c_window_metrics:
        c_window_metrics = [fm['metrics'] for fm in processed_frame_metrics if fm['metrics'] is not None]

    def get_contact_median(key):
        vals = [m[key] for m in c_window_metrics if key in m and m[key] is not None]
        return float(np.median(vals)) if len(vals) > 0 else None

    user_representative_metrics = {
        'front_elbow_angle': get_contact_median('front_elbow_angle'),
        'spine_lean': get_contact_median('spine_lean'),
        'head_knee_alignment': get_contact_median('head_knee_alignment'),
        'front_foot_direction': get_contact_median('front_foot_direction'),
        'shoulder_rotation': get_contact_median('shoulder_rotation'),
        'hip_rotation': get_contact_median('hip_rotation'),
        'stride_length': get_contact_median('stride_length')
    }

    # Follow-Through Metrics
    ft_min, ft_max = phases.get('follow_through', [min(total_frames, pose_impact_frame + 4), min(total_frames, pose_impact_frame + 20)])
    ft_metrics = [
        fm['metrics'] for fm in processed_frame_metrics
        if fm['metrics'] is not None and ft_min <= fm['frame_idx'] <= ft_max
    ]
    if ft_metrics:
        ft_angles = [m['front_elbow_angle'] for m in ft_metrics if 'front_elbow_angle' in m and m['front_elbow_angle'] is not None]
        ft_angle_val = float(np.mean(ft_angles)) if ft_angles else None
        ft_qual_val = float(np.clip((ft_angle_val / 150.0) * 100.0, 0.0, 100.0)) if ft_angle_val is not None else None
    else:
        ft_angle_val = None
        ft_qual_val = None

    user_representative_metrics['follow_through_angle'] = ft_angle_val
    user_representative_metrics['follow_through_quality'] = ft_qual_val

    # Head Stability
    ds_min = phases.get('downswing', [1, pose_impact_frame])[0]
    ds_max = phases.get('contact_window', [pose_impact_frame, pose_impact_frame])[1]
    ds_metrics = [
        fm['metrics'] for fm in processed_frame_metrics
        if fm['metrics'] is not None and ds_min <= fm['frame_idx'] <= ds_max
    ]
    if len(ds_metrics) > 1:
        n_xs = [m['nose_x_norm'] for m in ds_metrics if 'nose_x_norm' in m and m['nose_x_norm'] is not None]
        n_ys = [m['nose_y_norm'] for m in ds_metrics if 'nose_y_norm' in m and m['nose_y_norm'] is not None]
        head_stab_val = float(np.sqrt(np.var(n_xs) + np.var(n_ys))) if len(n_xs) > 1 else None
    else:
        head_stab_val = None

    user_representative_metrics['head_stability'] = head_stab_val

    # 9. Multi-Shot Evaluation via Correct Shot Baseline
    scoring_engine = MultiShotScoringEngine(
        shot_type=active_shot,
        config_path=config_path
    )
    evaluation_result = scoring_engine.evaluate(
        user_metrics=user_representative_metrics,
        handedness=handedness
    )

    # Bat and Ball Tracking quality summaries (Section 15, 16 & 28)
    bat_count = len(bat_detections)
    bat_coverage = round(bat_count / max(1, total_frames), 2)
    bat_continuity = "Good" if bat_coverage >= 0.60 else ("Fair" if bat_coverage >= 0.25 else "Poor")
    bat_tracking_status = "AVAILABLE" if bat_coverage >= 0.50 else ("PARTIAL" if bat_coverage >= 0.20 else "UNAVAILABLE")

    ball_count = len(ball_detections)
    ball_coverage = round(ball_count / max(1, total_frames), 2)
    ball_continuity = "Good" if ball_count >= 5 else ("Fair" if ball_count >= 2 else "Poor")
    ball_tracking_status = "AVAILABLE" if ball_count >= 4 else ("PARTIAL" if ball_count >= 2 else "UNAVAILABLE")

    valid_pose_count = sum(1 for lm in raw_landmarks_list if lm is not None)
    pose_quality = round(valid_pose_count / max(1, total_frames), 2)

    # Attach Unified Tracking & Quality summaries
    evaluation_result['bat_tracking'] = {
        'status': bat_tracking_status,
        'coverage_pct': round(bat_coverage * 100, 1),
        'detected_frames': bat_count,
        'continuity': bat_continuity,
        'confidence': round(float(np.mean([b.get('confidence', 0.8) for b in bat_detections.values()])), 2) if bat_detections else 0.0
    }
    evaluation_result['ball_tracking'] = {
        'status': ball_tracking_status,
        'coverage_pct': round(ball_coverage * 100, 1),
        'detected_frames': ball_count,
        'continuity': ball_continuity,
        'confidence': round(float(np.mean([b.get('confidence', 0.7) for b in ball_detections.values()])), 2) if ball_detections else 0.0
    }
    evaluation_result['quality'] = {
        'pose_quality': pose_quality,
        'bat_quality': bat_coverage,
        'ball_quality': ball_coverage
    }
    evaluation_result['impact_analysis'] = {
        'biomechanical_impact_frame': pose_impact_frame,
        'contact_window': phases.get('contact_window', [max(1, pose_impact_frame - 3), min(total_frames, pose_impact_frame + 3)])
    }

    # Attach Shot Analysis & Contact Analysis payloads matching unified specification (Section 29)
    is_unknown_shot = (active_shot == SHOT_UNKNOWN)
    evaluation_result['shot_analysis'] = {
        'predicted_shot': active_shot,
        'decision': classification_result.get('decision', 'UNKNOWN' if is_unknown_shot else 'CONFIDENT'),
        'confidence': round(classification_result['confidence'], 3),
        'confidence_level': classification_result.get('confidence_level', 'Low Confidence'),
        'top_candidate': classification_result.get('top_candidate', active_shot),
        'top_probability': round(classification_result.get('top_probability', classification_result['confidence']), 4),
        'top_class_margin': round(classification_result.get('top_class_margin', 0.0), 4),
        'reference_similarity': round(classification_result.get('reference_similarity', 0.0), 3),
        'feature_coverage': classification_result.get('feature_coverage', 1.0),
        'class_probabilities': {k: round(v, 4) for k, v in classification_result.get('class_probabilities', {}).items()},
        'class_scores': {k: round(v, 4) for k, v in classification_result.get('class_probabilities', {}).items()},
        'classification_status': classification_result.get('classification_status', 'unknown' if is_unknown_shot else 'classified'),
        'classification_reason': classification_result.get('classification_reason'),
        'status': 'unknown' if is_unknown_shot else 'classified',
        'feature_evidence': classification_result.get('feature_evidence', {}),
        'reason': classification_result.get('reason', '')
    }

    sigs = contact_analysis.get('signals', {})
    c_status_str = contact_analysis['status'].upper() if isinstance(contact_analysis['status'], str) else "UNKNOWN"
    if c_status_str == "CONTACT":
        c_status_str = "CONTACT_DETECTED"
    elif c_status_str == "NO_CONTACT":
        c_status_str = "NO_CONTACT_DETECTED"

    evaluation_result['contact_analysis'] = {
        'status': c_status_str,
        'confidence': round(contact_analysis.get('confidence', 41.0), 1),
        'confidence_score': round(contact_analysis.get('confidence', 41.0), 1),
        'evidence_score': round(contact_analysis.get('confidence', 41.0), 1),
        'biomechanical_impact_frame': pose_impact_frame,
        'pose_impact_frame': pose_impact_frame,
        'visual_contact_frame': contact_analysis.get('visual_contact_frame'),
        'contact_frame': contact_analysis.get('visual_contact_frame'),
        'estimated_frame': contact_analysis.get('estimated_frame'),
        'frame_difference': contact_analysis.get('frame_difference'),
        'bat_tracking_status': bat_tracking_status,
        'ball_tracking_status': ball_tracking_status,
        'bat_tracking_available': contact_analysis.get('bat_tracking_available', False),
        'ball_tracking_available': contact_analysis.get('ball_tracking_available', False),
        'evidence_coverage': contact_analysis.get('evidence_coverage', 0.0),
        'signals': {
            'spatial_proximity': sigs.get('spatial_proximity'),
            'temporal_alignment': sigs.get('temporal_alignment'),
            'relative_velocity': sigs.get('relative_velocity'),
            'ball_deflection': sigs.get('ball_deflection'),
            'bat_consistency': sigs.get('bat_consistency')
        },
        'reason': contact_analysis.get('reason', ''),
        'bat_ball_distance_px': sigs.get('bat_ball_distance_px'),
        'spatial_score': sigs.get('spatial_proximity'),
        'temporal_score': sigs.get('temporal_alignment'),
        'relative_velocity_score': sigs.get('relative_velocity'),
        'ball_trajectory_change_score': sigs.get('ball_deflection'),
        'bat_trajectory_consistency_score': sigs.get('bat_consistency'),
        'final_contact_score': contact_analysis.get('confidence')
    }

    # Backward compatibility mappings
    evaluation_result['metric_evaluations'] = {}
    if not is_unknown_shot:
        for mk, mv in evaluation_result.get('metric_scores', {}).items():
            evaluation_result['metric_evaluations'][mk] = {
                'name': mk.replace('_', ' ').title(),
                'user_value_display': f"{mv['raw_value']} {mv.get('unit', '')}" if mv['raw_value'] is not None else "N/A",
                'metric_score_display': f"{mv['score']} / 100" if mv['score'] is not None else "N/A",
                'metric_score': mv['score'],
                'assessment': mv['assessment'],
                'reference_mean': mv.get('reference_mean'),
                'reference_p10': mv.get('reference_p10'),
                'reference_p90': mv.get('reference_p90'),
                'reference_median': mv.get('reference_median'),
                'deviation_from_mean': round(abs(mv['raw_value'] - mv['reference_mean']), 2) if (mv['raw_value'] is not None and mv.get('reference_mean') is not None) else None
            }
    else:
        # Display general biomechanical measurements without scoring against any reference
        for mk, mv in evaluation_result.get('metric_scores', {}).items():
            raw_val = mv.get('raw_value')
            evaluation_result['metric_evaluations'][mk] = {
                'name': mk.replace('_', ' ').title(),
                'user_value_display': f"{raw_val} {mv.get('unit', '')}" if raw_val is not None else "N/A",
                'metric_score_display': "General Measurement Only",
                'metric_score': None,
                'assessment': mv.get('assessment', "General Biomechanical Measurement (Shot Unclassified - Scoring Withheld)"),
                'reference_mean': None,
                'reference_p10': None,
                'reference_p90': None,
                'reference_median': None,
                'deviation_from_mean': None
            }

    for cat_k, cat_v in evaluation_result.get('category_scores', {}).items():
        cat_v['category_name'] = cat_k.replace('_', ' ').title()
        cat_v['score_display'] = f"{cat_v['score']} / 100" if cat_v['score'] is not None else "N/A"
        cat_v['feedback'] = f"Category score {cat_v['score']}/100"

    # 10. Generate Output Video with Rich Visual Annotations (Section 23)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (frame_width, frame_height))

    pred_shot_key = classification_result.get('predicted_shot', 'unknown')
    shot_name_disp = SHOT_DISPLAY_NAMES.get(pred_shot_key, pred_shot_key.replace('_', ' ')).upper()
    shot_label_text = f"SHOT: {shot_name_disp}"
    shot_conf_text = f"CONFIDENCE: {round(classification_result['confidence'] * 100, 1)}%"

    contact_status_raw = str(contact_analysis.get('status', 'unknown')).lower()
    contact_conf_val = round(contact_analysis.get('confidence', 41.0), 1)
    v_contact_frame = contact_analysis.get('visual_contact_frame') or contact_analysis.get('estimated_frame')

    bat_history = []
    ball_history = []

    for f_idx, frame in enumerate(cached_frames, 1):
        ann = frame.copy()
        lm = raw_landmarks_list[f_idx - 1] if f_idx <= len(raw_landmarks_list) else None

        # Draw batter skeleton
        if lm:
            def ppx(name): return (int(lm[name][0] * frame_width), int(lm[name][1] * frame_height))
            connections = [
                ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'), ('left_ankle', 'left_foot_index'), ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle'), ('right_ankle', 'right_foot_index')
            ]
            for p1, p2 in connections:
                cv2.line(ann, ppx(p1), ppx(p2), (245, 117, 66), 2)
            for k in lm:
                if not k.startswith('_'):
                    cv2.circle(ann, ppx(k), 4, (0, 255, 255), -1)

        # Record Bat Tracking History
        if f_idx in bat_detections:
            b_info = bat_detections[f_idx]
            bc = b_info.get('center', (0, 0))
            is_est = b_info.get('is_estimated', False)
            bat_history.append((int(bc[0]), int(bc[1]), is_est, f_idx))

        # Record Ball Tracking History
        if f_idx in ball_detections:
            bl_info = ball_detections[f_idx]
            bl_c = bl_info.get('center', (0, 0))
            ball_history.append((int(bl_c[0]), int(bl_c[1]), f_idx))

        # Render Bat Trajectory Trail (Past 18 frames)
        recent_bat_pts = [p for p in bat_history if 0 <= (f_idx - p[3]) <= 18]
        for i in range(1, len(recent_bat_pts)):
            pt1 = (recent_bat_pts[i-1][0], recent_bat_pts[i-1][1])
            pt2 = (recent_bat_pts[i][0], recent_bat_pts[i][1])
            col = (0, 165, 255) if recent_bat_pts[i][2] else (0, 230, 255)
            cv2.line(ann, pt1, pt2, col, 2, cv2.LINE_AA)

        # Render Ball Trajectory Trail (Past 18 frames)
        recent_ball_pts = [p for p in ball_history if 0 <= (f_idx - p[2]) <= 18]
        for i in range(1, len(recent_ball_pts)):
            pt1 = (recent_ball_pts[i-1][0], recent_ball_pts[i-1][1])
            pt2 = (recent_ball_pts[i][0], recent_ball_pts[i][1])
            cv2.line(ann, pt1, pt2, (255, 60, 220), 2, cv2.LINE_AA)

        # Highlight Current Frame Bat
        if f_idx in bat_detections:
            bat = bat_detections[f_idx]
            bx, by, bw, bh = bat['bbox']
            is_est = bat.get('is_estimated', False)
            bat_color = (0, 165, 255) if is_est else (255, 180, 0)

            # Bounding box
            cv2.rectangle(ann, (bx, by), (bx + bw, by + bh), bat_color, 2)

            # Centerline segment
            if 'segment' in bat:
                cv2.line(ann, tuple(bat['segment'][0]), tuple(bat['segment'][1]), (0, 220, 255), 3)

            # Bat center and tip
            cx, cy = bat['center']
            cv2.circle(ann, (int(cx), int(cy)), 5, (0, 255, 255), -1)
            tip = bat.get('tip')
            if tip:
                cv2.circle(ann, (int(tip[0]), int(tip[1])), 6, (0, 140, 255), -1)
                cv2.putText(ann, "TIP", (int(tip[0]) + 6, int(tip[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

            # Trajectory & Direction Overlay
            if is_est:
                cv2.putText(ann, "BAT (ESTIMATED)", (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1)
            else:
                cv2.putText(ann, "BAT ------>", (bx, max(20, by - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2)
                cv2.putText(ann, "trajectory", (bx, max(32, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1)

            # Direction swing arrow
            if len(recent_bat_pts) >= 4:
                p_old = (recent_bat_pts[-4][0], recent_bat_pts[-4][1])
                p_cur = (recent_bat_pts[-1][0], recent_bat_pts[-1][1])
                if (p_cur[0] - p_old[0])**2 + (p_cur[1] - p_old[1])**2 > 25:
                    cv2.arrowedLine(ann, p_old, p_cur, (0, 255, 255), 2, tipLength=0.3)

            # Candidate contact region around bat tip during candidate window
            if abs(f_idx - pose_impact_frame) <= 6:
                zone_center = (int(tip[0]), int(tip[1])) if tip else (int(cx), int(cy))
                cv2.circle(ann, zone_center, 30, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(ann, "CONTACT ZONE", (zone_center[0] - 45, zone_center[1] - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Highlight Current Frame Ball
        if f_idx in ball_detections:
            ball = ball_detections[f_idx]
            b_center = (int(ball['center'][0]), int(ball['center'][1]))
            cv2.circle(ann, b_center, int(ball['radius']) + 3, (0, 0, 255), 2)
            cv2.circle(ann, b_center, 2, (0, 0, 255), -1)
            cv2.putText(ann, "BALL", (b_center[0] + 6, b_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Shot Classification Banner (Top-Left)
        cv2.rectangle(ann, (10, 10), (340, 80), (15, 15, 15), -1)
        cv2.rectangle(ann, (10, 10), (340, 80), (70, 70, 70), 1)
        cv2.putText(ann, shot_label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 200), 2)
        cv2.putText(ann, shot_conf_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)

        # Contact Status Display (Top-Right)
        cv2.rectangle(ann, (frame_width - 360, 10), (frame_width - 10, 85), (15, 15, 15), -1)
        cv2.rectangle(ann, (frame_width - 360, 10), (frame_width - 10, 85), (70, 70, 70), 1)
        if contact_status_raw in ("contact", "contact_detected"):
            cv2.putText(ann, "CONTACT: DETECTED", (frame_width - 345, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)
            cv2.putText(ann, f"CONFIDENCE: {contact_conf_val}%", (frame_width - 345, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(ann, f"CONTACT FRAME: {v_contact_frame}", (frame_width - 345, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1)
        elif contact_status_raw in ("no_contact", "no_contact_detected"):
            cv2.putText(ann, "CONTACT: NO CONTACT", (frame_width - 345, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 100, 255), 2)
            cv2.putText(ann, f"CONFIDENCE: {contact_conf_val}%", (frame_width - 345, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        else:
            cv2.putText(ann, "CONTACT: UNKNOWN", (frame_width - 345, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 180, 255), 2)
            cv2.putText(ann, f"CONFIDENCE: {contact_conf_val}%", (frame_width - 345, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Distinct Biomechanical Impact vs Visual Contact Markers
        if abs(f_idx - pose_impact_frame) <= 1:
            cv2.putText(ann, f"[POSE IMPACT ESTIMATE: FRAME {pose_impact_frame}]", (frame_width // 2 - 200, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)

        if contact_status_raw in ("contact", "contact_detected") and v_contact_frame and abs(f_idx - v_contact_frame) <= 2:
            cv2.putText(ann, f"[BAT-BALL CONTACT: FRAME {v_contact_frame}]", (frame_width // 2 - 180, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
            # Draw contact star marker at contact location
            c_loc = None
            if f_idx in ball_detections:
                c_loc = (int(ball_detections[f_idx]['center'][0]), int(ball_detections[f_idx]['center'][1]))
            elif f_idx in bat_detections and bat_detections[f_idx].get('tip'):
                c_loc = (int(bat_detections[f_idx]['tip'][0]), int(bat_detections[f_idx]['tip'][1]))
            if c_loc:
                cv2.drawMarker(ann, c_loc, (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=24, thickness=2)

        out.write(ann)

    out.release()

    # 11. Save output/evaluation.json, report.txt, and temporal smoothness plot
    evaluation_json_path = 'output/evaluation.json'
    evaluation_report_path = 'output/evaluation_report.txt'
    temporal_smoothness_path = 'output/temporal_smoothness.png'

    with open(evaluation_json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, indent=4)

    # Save temporal smoothness plot
    if elbow_angles and spine_leans:
        plt.figure(figsize=(10, 5))
        plt.plot(elbow_angles, label='Elbow Angle (°)', color='#1f77b4')
        plt.plot(spine_leans, label='Spine Lean (°)', color='#ff7f0e')
        plt.axvline(x=pose_impact_frame, color='red', linestyle='--', label=f'Pose Impact (F:{pose_impact_frame})')
        if v_contact_frame:
            plt.axvline(x=v_contact_frame, color='green', linestyle=':', label=f'Visual Contact (F:{v_contact_frame})')
        plt.title(f'Temporal Smoothness — {SHOT_DISPLAY_NAMES.get(active_shot, active_shot.title())}')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(temporal_smoothness_path)
        plt.close()

    # Generate and save formatted text report
    text_report = MultiShotReportGenerator.generate_text_report(evaluation_result)
    with open(evaluation_report_path, 'w', encoding='utf-8') as f:
        f.write(text_report)

    return evaluation_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AthleteRise Multi-Shot Batting Analysis")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to input batting video")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to configuration JSON")
    parser.add_argument("--shot", "-s", type=str, default=None, help="Override shot type (cover_drive, straight_drive, pull_shot)")
    parser.add_argument("--no-display", action="store_true", help="Run without UI display")
    args = parser.parse_args()

    main(video_input=args.video, config_input=args.config, shot_type_override=args.shot)
