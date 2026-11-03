"""
AthleteRise - Generic Multi-Shot Reference Calibration Engine
Processes reference video datasets independently per shot class:
- Cover Drive (data/reference_videos/cover_drive)
- Straight Drive (data/reference_videos/straight_drive)
- Pull Shot (data/reference_videos/pull_shot)
Derives comprehensive empirical statistics: Mean, Median, Std, Variance, Min, Max,
P10, P25, P50, P75, P90, IQR, Valid sample count, and Outlier diagnostics.
"""

import os
import json
import csv
import glob
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# MediaPipe pose import handling with safe fallback
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
from src.biomechanics.metrics import BiomechanicsCalculator, detect_handedness
from src.phase_detection.phase_factory import PhaseDetectorFactory
from src.report.calibration_reporter import CoverDriveCalibrationReporter
from src.shot_analysis.shot_features import ShotFeatureExtractor

METRIC_KEYS = [
    'front_elbow_angle',
    'spine_lean',
    'head_knee_alignment',
    'front_foot_direction',
    'shoulder_rotation',
    'hip_rotation',
    'stride_length',
    'head_stability',
    'follow_through_angle',
    'follow_through_quality'
]

CALIBRATION_FEATURE_KEYS = METRIC_KEYS + [
    'rotary_shoulder_diff',
    'wrist_horizontal_travel',
    'wrist_vertical_travel'
]

METRIC_DISPLAY_NAMES = {
    'front_elbow_angle': 'Front Elbow Angle (°)',
    'spine_lean': 'Spine Lean Tilt (°)',
    'head_knee_alignment': 'Head-Knee Alignment (norm)',
    'front_foot_direction': 'Front Foot Direction (°)',
    'shoulder_rotation': 'Shoulder Rotation (°)',
    'hip_rotation': 'Hip Rotation (°)',
    'stride_length': 'Stride Length (norm)',
    'head_stability': 'Head Stability (norm)',
    'follow_through_angle': 'Follow-Through Angle (°)',
    'follow_through_quality': 'Follow-Through Quality (0-100)',
    'rotary_shoulder_diff': 'Rotary Shoulder Diff (°)',
    'wrist_horizontal_travel': 'Wrist Horizontal Travel (norm)',
    'wrist_vertical_travel': 'Wrist Vertical Travel (norm)'
}

class ShotCalibrationEngine:
    """
    Generic Reference Calibration Engine for any cricket shot class.
    Processes dedicated reference videos and exports statistics database.
    """
    def __init__(self, shot_type: str = "cover_drive", reference_dir: Optional[str] = None, output_dir: Optional[str] = None, stance_override: Optional[str] = None):
        self.shot_type = shot_type.lower().replace(" ", "_").strip()
        self.reference_dir = Path(reference_dir or f"data/reference_videos/{self.shot_type}")
        self.output_dir = Path(output_dir or f"output/reference_calibration/{self.shot_type}")
        self.stance_override = stance_override
        self.mp_pose = mp_pose

        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def process_single_video(self, video_path: str) -> Dict[str, Any]:
        """
        Ingests, validates, and processes a single reference video clip.
        """
        filename = os.path.basename(video_path)
        res = {
            'video_id': Path(filename).stem,
            'filename': filename,
            'filepath': video_path,
            'status': 'valid',
            'exclusion_reason': '',
            'duration_sec': 0.0,
            'fps': 25,
            'total_frames': 0,
            'resolution': '0x0',
            'valid_pose_frames': 0,
            'pose_detection_pct': 0.0,
            'contact_frame': 1,
            'contact_detected': False,
            'handedness': 'right',
            'metrics': {}
        }

        if not os.path.exists(video_path):
            res['status'] = 'excluded'
            res['exclusion_reason'] = 'File not found'
            return res

        try:
            preprocessor = VideoPreprocessor(video_path=video_path, save_frames_dir=None)
        except Exception as e:
            res['status'] = 'excluded'
            res['exclusion_reason'] = f"Failed to open video: {str(e)}"
            return res

        res['fps'] = preprocessor.fps
        res['total_frames'] = preprocessor.total_frames
        res['duration_sec'] = round(preprocessor.total_frames / (preprocessor.fps or 25), 2)
        res['resolution'] = f"{preprocessor.width}x{preprocessor.height}"

        if preprocessor.total_frames < 5:
            res['status'] = 'excluded'
            res['exclusion_reason'] = 'Video too short (< 5 frames)'
            preprocessor.release()
            return res

        from src.pose_estimation.posture_tracker import VisionPostureTracker
        tracker = VisionPostureTracker(self.mp_pose)
        tracker.profile_video(video_path)

        raw_landmarks_list = []
        frame_metrics_list = []

        for frame_idx, frame in preprocessor.read_frames():
            lm_dict = tracker.process_frame(frame, frame_idx, res['total_frames'])
            raw_landmarks_list.append(lm_dict)
            frame_metrics_list.append({
                'frame_idx': frame_idx,
                'landmarks': lm_dict
            })

        preprocessor.release()
        tracker.close()

        valid_frames_count = sum(1 for lm in raw_landmarks_list if lm is not None)
        res['valid_pose_frames'] = valid_frames_count
        res['pose_detection_pct'] = round((valid_frames_count / max(1, res['total_frames'])) * 100, 1)

        # Determine Handedness
        handedness = self.stance_override or detect_handedness(raw_landmarks_list)
        res['handedness'] = handedness

        calculator = BiomechanicsCalculator(handedness=handedness, frame_width=preprocessor.width, frame_height=preprocessor.height)

        # Compute instantaneous metrics
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
            processed_frame_metrics.append({
                'frame_idx': fm['frame_idx'],
                'metrics': m,
                'landmarks': lm
            })

        # Shot-Specific Phase & Contact Frame Detection
        phase_detector = PhaseDetectorFactory.create(self.shot_type)
        contact_frame = phase_detector.detect_contact_frame(processed_frame_metrics, res['total_frames'])
        res['contact_frame'] = contact_frame
        res['contact_detected'] = True

        phases = phase_detector.segment_phases(processed_frame_metrics, contact_frame, res['total_frames'])

        # Extract features using unified ShotFeatureExtractor (guaranteeing exact schema parity)
        extractor = ShotFeatureExtractor(handedness=handedness)
        feat_dict = extractor.extract_features(processed_frame_metrics, contact_frame, res['total_frames'])

        ft_angle = feat_dict.get('follow_through_angle')
        ft_qual = float(np.clip((ft_angle / 150.0) * 100.0, 0.0, 100.0)) if ft_angle is not None else None

        res['metrics'] = {
            'front_elbow_angle': feat_dict.get('front_elbow_angle'),
            'spine_lean': feat_dict.get('spine_lean'),
            'head_knee_alignment': feat_dict.get('head_knee_alignment'),
            'front_foot_direction': feat_dict.get('front_foot_direction'),
            'shoulder_rotation': feat_dict.get('shoulder_rotation'),
            'hip_rotation': feat_dict.get('hip_rotation'),
            'stride_length': feat_dict.get('stride_length'),
            'head_stability': feat_dict.get('head_stability'),
            'follow_through_angle': ft_angle,
            'follow_through_quality': round(ft_qual, 2) if ft_qual is not None else None,
            'rotary_shoulder_diff': feat_dict.get('rotary_shoulder_diff'),
            'wrist_horizontal_travel': feat_dict.get('wrist_horizontal_travel'),
            'wrist_vertical_travel': feat_dict.get('wrist_vertical_travel')
        }

        return res

    def run_calibration(self, video_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executes reference calibration across all video paths for this shot type.
        Calculates all 14 required statistical descriptors per metric.
        """
        if video_paths is None:
            video_paths = sorted(glob.glob(str(self.reference_dir / "*.avi")) + glob.glob(str(self.reference_dir / "*.mp4")))

        results_per_video = []
        for vpath in video_paths:
            v_res = self.process_single_video(vpath)
            results_per_video.append(v_res)

        valid_videos = [v for v in results_per_video if v['status'] in ('valid', 'warning')]
        excluded_videos = [v for v in results_per_video if v['status'] == 'excluded']
        sample_count = len(valid_videos)

        statistics = {}
        outlier_diagnostics = {}

        for key in CALIBRATION_FEATURE_KEYS:
            vals = [v['metrics'][key] for v in valid_videos if key in v['metrics'] and v['metrics'][key] is not None]
            if len(vals) > 0:
                mean_val = float(np.mean(vals))
                median_val = float(np.median(vals))
                std_val = float(np.std(vals))
                var_val = float(np.var(vals))
                min_val = float(np.min(vals))
                max_val = float(np.max(vals))
                p10_val = float(np.percentile(vals, 10))
                p25_val = float(np.percentile(vals, 25))
                p50_val = median_val
                p75_val = float(np.percentile(vals, 75))
                p90_val = float(np.percentile(vals, 90))
                iqr_val = float(p75_val - p25_val)

                lower_bound = p25_val - 1.5 * iqr_val
                upper_bound = p75_val + 1.5 * iqr_val

                outliers = []
                for v in valid_videos:
                    val = v['metrics'].get(key)
                    if val is not None and (val < lower_bound or val > upper_bound):
                        outliers.append({
                            'video_id': v['video_id'],
                            'filename': v['filename'],
                            'raw_value': val,
                            'reason': f"Outside IQR bounds [{round(lower_bound, 2)}, {round(upper_bound, 2)}]"
                        })

                statistics[key] = {
                    'name': METRIC_DISPLAY_NAMES[key],
                    'valid_samples_count': len(vals),
                    'mean': round(mean_val, 4),
                    'median': round(median_val, 4),
                    'std': round(std_val, 4),
                    'variance': round(var_val, 4),
                    'min': round(min_val, 4),
                    'max': round(max_val, 4),
                    'q1': round(p25_val, 4),
                    'q3': round(p75_val, 4),
                    'p10': round(p10_val, 4),
                    'p25': round(p25_val, 4),
                    'p50': round(p50_val, 4),
                    'p75': round(p75_val, 4),
                    'p90': round(p90_val, 4),
                    'iqr': round(iqr_val, 4),
                    'outliers': outliers,
                    'reference_range': {
                        'lower': round(p10_val, 4),
                        'upper': round(p90_val, 4),
                        'method': 'P10-P90'
                    }
                }
                outlier_diagnostics[key] = outliers
            else:
                statistics[key] = {
                    'name': METRIC_DISPLAY_NAMES.get(key, key),
                    'valid_samples_count': 0,
                    'mean': None, 'median': None, 'std': None, 'variance': None,
                    'min': None, 'max': None, 'q1': None, 'q3': None,
                    'p10': None, 'p25': None, 'p50': None,
                    'p75': None, 'p90': None, 'iqr': None, 'outliers': [],
                    'reference_range': {'lower': None, 'upper': None, 'method': 'P10-P90'}
                }
                outlier_diagnostics[key] = []

        handedness_counts = {'right': 0, 'left': 0}
        for v in valid_videos:
            h = v.get('handedness', 'right')
            handedness_counts[h] = handedness_counts.get(h, 0) + 1
        handedness_summary = 'mixed' if (handedness_counts['right'] > 0 and handedness_counts['left'] > 0) else ('left' if handedness_counts['left'] > 0 else 'right')

        reference_data = {
            'shot_type': self.shot_type,
            'reference_status': 'provisional',
            'data_source_label': 'Calculated from available reference/calibration data across all reference videos',
            'sample_count': sample_count,
            'total_videos_processed': len(results_per_video),
            'valid_videos_count': sample_count,
            'excluded_videos_count': len(excluded_videos),
            'handedness': handedness_summary,
            'handedness_distribution': handedness_counts,
            'statistics': statistics,
            'outlier_diagnostics': outlier_diagnostics,
            'per_video_data': results_per_video
        }

        # Save JSON output
        json_path = self.output_dir / f"{self.shot_type}_reference.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(reference_data, f, indent=4)

        # Save CSV outputs with full statistical descriptors: Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR, P10, P90
        stats_csv_path = self.output_dir / f"{self.shot_type}_statistics.csv"
        with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'name', 'valid_samples', 'mean', 'median', 'std', 'variance', 'min', 'max', 'q1', 'q3', 'iqr', 'p10', 'p90'])
            for k, s in statistics.items():
                writer.writerow([k, s['name'], s['valid_samples_count'], s['mean'], s['median'], s['std'], s.get('variance'), s['min'], s['max'], s.get('q1'), s.get('q3'), s['iqr'], s['p10'], s['p90']])

        metrics_csv_path = self.output_dir / f"{self.shot_type}_video_metrics.csv"
        with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            export_keys = CALIBRATION_FEATURE_KEYS
            headers = ['video_id', 'filename', 'status', 'handedness'] + export_keys
            writer.writerow(headers)
            for v in results_per_video:
                row = [v['video_id'], v['filename'], v['status'], v.get('handedness', 'right')]
                for mk in export_keys:
                    row.append(v['metrics'].get(mk, ''))
                writer.writerow(row)

        # Automatically generate comprehensive box plots across all reference videos
        try:
            from src.calibration.biomechanical_statistics import BiomechanicalStatisticsAnalyzer
            analyzer = BiomechanicalStatisticsAnalyzer(reference_data, shot_type=self.shot_type)
            box_plot_path = self.output_dir / f"{self.shot_type}_boxplots.png"
            analyzer.plot_all_metric_boxplots(save_path=box_plot_path)
        except Exception as e:
            print(f"[!] Warning: Could not generate box plot figure: {e}")

        # Automatically generate reference distribution dashboard and HTML report
        try:
            from src.report.calibration_reporter import ShotCalibrationReporter
            reporter = ShotCalibrationReporter(reference_data, output_dir=self.output_dir, shot_type=self.shot_type)
            reporter.generate_distribution_plots()
            reporter.generate_html_report()
        except Exception as e:
            print(f"[!] Warning: Could not generate distribution plots or HTML report: {e}")

        return reference_data
