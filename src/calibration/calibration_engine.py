import os
import json
import csv
import glob
import cv2
import numpy as np
from pathlib import Path

# Robust MediaPipe pose import across Python environments
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except Exception:
    try:
        import mediapipe.python.solutions.pose as mp_pose
        import mediapipe.python.solutions.drawing_utils as mp_drawing
    except Exception:
        mp_pose = None
        mp_drawing = None

from src.preprocessing.video_preprocessing import VideoPreprocessor
from src.biomechanics.metrics import BiomechanicsCalculator, detect_handedness
from src.phase_detection.phase_detector import CoverDrivePhaseDetector

METRIC_KEYS = [
    'front_elbow_angle',
    'spine_lean',
    'head_knee_alignment',
    'front_foot_direction',
    'follow_through_angle',
    'follow_through_quality',
    'head_stability',
    'shoulder_rotation',
    'hip_rotation',
    'stride_length'
]

METRIC_DISPLAY_NAMES = {
    'front_elbow_angle': 'Front Elbow Angle (°)',
    'spine_lean': 'Spine Lean Tilt (°)',
    'head_knee_alignment': 'Head-Knee Alignment (norm)',
    'front_foot_direction': 'Front Foot Direction (°)',
    'follow_through_angle': 'Follow-Through Angle (°)',
    'follow_through_quality': 'Follow-Through Quality',
    'head_stability': 'Head Stability (norm)',
    'shoulder_rotation': 'Shoulder Rotation (°)',
    'hip_rotation': 'Hip Rotation (°)',
    'stride_length': 'Stride Length (norm)'
}

def get_dedicated_reference_videos(dataset_dir="data/reference_videos/cover_drive"):
    """
    Scans ONLY the dedicated reference folder for Cover Drive calibration videos (.mp4, .avi).
    Does NOT scan project root or test directories.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        dataset_path.mkdir(parents=True, exist_ok=True)
        return []

    files = list(dataset_path.glob("*.mp4")) + list(dataset_path.glob("*.avi")) + list(dataset_path.glob("*.mov"))
    valid_files = [str(f) for f in files if not f.name.startswith("temp") and not f.name.startswith(".")]
    valid_files.sort()
    return valid_files

class CoverDriveCalibrationEngine:
    """
    Statistical Reference Calibration Pipeline for Cover Drive Shots.
    Processes reference videos strictly from data/reference_videos/cover_drive/,
    extracts robust per-video representative metrics, calculates 10 statistical descriptors,
    and exports JSON/CSV baseline databases. Strictly excludes None/invalid measurements.
    """
    def __init__(self, output_dir="output/reference_calibration/cover_drive", stance_override=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stance_override = stance_override
        self.mp_pose = mp_pose

    def process_single_video(self, video_path):
        """
        Validates, processes, and extracts representative metrics for a single Cover Drive video.
        """
        video_path = str(video_path)
        filename = os.path.basename(video_path)

        res = {
            'video_id': Path(filename).stem,
            'filename': filename,
            'filepath': video_path,
            'status': 'valid',
            'exclusion_reason': '',
            'duration_sec': 0.0,
            'fps': 30,
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
        res['duration_sec'] = round(preprocessor.total_frames / (preprocessor.fps or 30), 2)
        res['resolution'] = f"{preprocessor.width}x{preprocessor.height}"

        if preprocessor.total_frames < 5:
            res['status'] = 'excluded'
            res['exclusion_reason'] = 'Video too short (< 5 frames)'
            preprocessor.release()
            return res

        # Process frames with MediaPipe Pose
        pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        raw_landmarks_list = []
        frame_metrics_list = []

        calculator_dummy = BiomechanicsCalculator(handedness='right', frame_width=preprocessor.width, frame_height=preprocessor.height)

        for frame_idx, frame in preprocessor.read_frames():
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results and results.pose_landmarks:
                lm_dict = calculator_dummy.extract_frame_landmarks(results.pose_landmarks, self.mp_pose)
                raw_landmarks_list.append(lm_dict)
                frame_metrics_list.append({
                    'frame_idx': frame_idx,
                    'landmarks': lm_dict
                })
            else:
                raw_landmarks_list.append(None)
                frame_metrics_list.append({
                    'frame_idx': frame_idx,
                    'landmarks': None
                })

        preprocessor.release()
        pose.close()

        valid_frames_count = sum(1 for lm in raw_landmarks_list if lm is not None)
        res['valid_pose_frames'] = valid_frames_count
        res['pose_detection_pct'] = round((valid_frames_count / max(1, res['total_frames'])) * 100, 1)

        if valid_frames_count == 0:
            res['status'] = 'excluded'
            res['exclusion_reason'] = 'No pose detected in video'
            return res

        if res['pose_detection_pct'] < 50.0:
            res['status'] = 'warning'
            res['exclusion_reason'] = f"Low pose detection rate ({res['pose_detection_pct']}%)"

        # Determine Handedness
        handedness = self.stance_override or detect_handedness(raw_landmarks_list)
        res['handedness'] = handedness

        # Initialize Handedness-specific Calculator
        calculator = BiomechanicsCalculator(handedness=handedness, frame_width=preprocessor.width, frame_height=preprocessor.height)

        # Compute Instantaneous Frame Metrics
        processed_frame_metrics = []
        for fm in frame_metrics_list:
            if fm['landmarks'] is not None:
                m = calculator.compute_frame_metrics(fm['landmarks'])
            else:
                m = None
            processed_frame_metrics.append({
                'frame_idx': fm['frame_idx'],
                'metrics': m
            })

        # Phase & Contact Frame Detection
        phase_detector = CoverDrivePhaseDetector()
        contact_frame = phase_detector.detect_contact_frame(processed_frame_metrics, res['total_frames'])
        res['contact_frame'] = contact_frame
        res['contact_detected'] = True

        phases = phase_detector.segment_phases(processed_frame_metrics, contact_frame, res['total_frames'])

        # Aggregate Representative Video Metrics (Contact Window & Phase Medians)
        c_min, c_max = phases['contact_window']
        contact_window_metrics = [
            fm['metrics'] for fm in processed_frame_metrics
            if fm['metrics'] is not None and c_min <= fm['frame_idx'] <= c_max
        ]

        if not contact_window_metrics:
            contact_window_metrics = [fm['metrics'] for fm in processed_frame_metrics if fm['metrics'] is not None]

        # Extract representative values (medians around contact window), strictly excluding None/invalid values
        def get_contact_median(key):
            vals = [m[key] for m in contact_window_metrics if key in m and m[key] is not None]
            return float(np.median(vals)) if len(vals) > 0 else None

        front_elbow_angle = get_contact_median('front_elbow_angle')
        spine_lean = get_contact_median('spine_lean')
        head_knee_align = get_contact_median('head_knee_alignment')
        front_foot_dir = get_contact_median('front_foot_direction')
        shoulder_rot = get_contact_median('shoulder_rotation')
        hip_rot = get_contact_median('hip_rotation')
        stride_len = get_contact_median('stride_length')

        # Follow-Through Metrics
        ft_min, ft_max = phases['follow_through']
        ft_metrics = [
            fm['metrics'] for fm in processed_frame_metrics
            if fm['metrics'] is not None and ft_min <= fm['frame_idx'] <= ft_max
        ]

        if ft_metrics:
            ft_elbow_angles = [m['front_elbow_angle'] for m in ft_metrics if m['front_elbow_angle'] is not None]
            follow_through_angle = float(np.mean(ft_elbow_angles)) if ft_elbow_angles else front_elbow_angle
            follow_through_quality = float(np.clip((follow_through_angle / 150.0) * 100, 0, 100)) if follow_through_angle is not None else None
        else:
            follow_through_angle = front_elbow_angle
            follow_through_quality = float(np.clip((follow_through_angle / 150.0) * 100, 0, 100)) if follow_through_angle is not None else None

        # Head Stability (Normalized by Shoulder Width)
        ds_min = phases['downswing'][0]
        ds_max = phases['contact_window'][1]
        ds_metrics = [
            fm['metrics'] for fm in processed_frame_metrics
            if fm['metrics'] is not None and ds_min <= fm['frame_idx'] <= ds_max
        ]

        if len(ds_metrics) > 1:
            nose_xs_norm = [m['nose_x_norm'] for m in ds_metrics if 'nose_x_norm' in m and m['nose_x_norm'] is not None]
            nose_ys_norm = [m['nose_y_norm'] for m in ds_metrics if 'nose_y_norm' in m and m['nose_y_norm'] is not None]
            if len(nose_xs_norm) > 1:
                head_stability = float(np.sqrt(np.var(nose_xs_norm) + np.var(nose_ys_norm)))
            else:
                head_stability = None
        else:
            head_stability = None

        res['metrics'] = {
            'front_elbow_angle': round(front_elbow_angle, 2) if front_elbow_angle is not None else None,
            'spine_lean': round(spine_lean, 2) if spine_lean is not None else None,
            'head_knee_alignment': round(head_knee_align, 4) if head_knee_align is not None else None,
            'front_foot_direction': round(front_foot_dir, 2) if front_foot_dir is not None else None,
            'follow_through_angle': round(follow_through_angle, 2) if follow_through_angle is not None else None,
            'follow_through_quality': round(follow_through_quality, 2) if follow_through_quality is not None else None,
            'head_stability': round(head_stability, 4) if head_stability is not None else None,
            'shoulder_rotation': round(shoulder_rot, 2) if shoulder_rot is not None else None,
            'hip_rotation': round(hip_rot, 2) if hip_rot is not None else None,
            'stride_length': round(stride_len, 4) if stride_len is not None else None
        }

        return res

    def run_calibration(self, video_paths):
        """
        Executes calibration across input Cover Drive videos.
        Calculates 10 statistical descriptors, reference ranges, and outlier flags.
        Excludes None/missing measurements strictly from statistics.
        Exports cover_drive_reference.json, cover_drive_video_metrics.csv, and cover_drive_statistics.csv.
        """
        results_per_video = []

        for vpath in video_paths:
            v_res = self.process_single_video(vpath)
            results_per_video.append(v_res)

        valid_videos = [v for v in results_per_video if v['status'] in ('valid', 'warning')]
        excluded_videos = [v for v in results_per_video if v['status'] == 'excluded']

        sample_count = len(valid_videos)
        ref_status = "provisional"

        statistics = {}
        outlier_diagnostics = {}

        for key in METRIC_KEYS:
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
                p75_val = float(np.percentile(vals, 75))
                p90_val = float(np.percentile(vals, 90))

                iqr = p75_val - p25_val
                lower_outlier_bound = p25_val - 1.5 * iqr
                upper_outlier_bound = p75_val + 1.5 * iqr

                outliers = []
                for v in valid_videos:
                    val = v['metrics'].get(key, None)
                    if val is not None and (val < lower_outlier_bound or val > upper_outlier_bound):
                        outliers.append({
                            'video_id': v['video_id'],
                            'filename': v['filename'],
                            'raw_value': val,
                            'reason': f"Outside IQR bounds [{round(lower_outlier_bound, 2)}, {round(upper_outlier_bound, 2)}]"
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
                    'iqr': round(iqr, 4),
                    'p10': round(p10_val, 4),
                    'p25': round(p25_val, 4),
                    'p50': round(median_val, 4),
                    'p75': round(p75_val, 4),
                    'p90': round(p90_val, 4),
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
                    'min': None, 'max': None, 'q1': None, 'q3': None, 'iqr': None,
                    'p10': None, 'p25': None, 'p50': None, 'p75': None, 'p90': None,
                    'reference_range': {'lower': None, 'upper': None, 'method': 'P10-P90'}
                }
                outlier_diagnostics[key] = []

        handedness_counts = {'right': 0, 'left': 0}
        for v in valid_videos:
            h = v.get('handedness', 'right')
            handedness_counts[h] = handedness_counts.get(h, 0) + 1

        handedness_summary = 'mixed' if (handedness_counts['right'] > 0 and handedness_counts['left'] > 0) else ('left' if handedness_counts['left'] > 0 else 'right')

        reference_data = {
            'shot_type': 'cover_drive',
            'reference_status': ref_status,
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
        json_path = self.output_dir / "cover_drive_reference.json"
        with open(json_path, 'w') as f:
            json.dump(reference_data, f, indent=4)

        # Save Per-Video Metrics CSV (Writing N/A for None values)
        csv_video_path = self.output_dir / "cover_drive_video_metrics.csv"
        with open(csv_video_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['video_id', 'filename', 'status', 'handedness', 'duration_sec', 'fps', 'total_frames', 'resolution', 'pose_detection_pct', 'contact_frame'] + METRIC_KEYS
            writer.writerow(header)

            for v in results_per_video:
                row = [
                    v['video_id'], v['filename'], v['status'], v['handedness'],
                    v['duration_sec'], v['fps'], v['total_frames'], v['resolution'],
                    v['pose_detection_pct'], v['contact_frame']
                ] + [v['metrics'].get(k, "N/A") if v['metrics'].get(k) is not None else "N/A" for k in METRIC_KEYS]
                writer.writerow(row)

        # Save Statistics CSV with all 11 required descriptors: Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR, P10, P90
        csv_stats_path = self.output_dir / "cover_drive_statistics.csv"
        with open(csv_stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric_key', 'metric_name', 'valid_samples_count', 'mean', 'median', 'std', 'variance', 'min', 'max', 'q1', 'q3', 'iqr', 'p10', 'p90', 'ref_lower_p10', 'ref_upper_p90'])

            for k in METRIC_KEYS:
                st = statistics[k]
                ref_low = st['reference_range']['lower']
                ref_high = st['reference_range']['upper']
                writer.writerow([
                    k, st['name'], st['valid_samples_count'],
                    st['mean'] if st['mean'] is not None else "N/A",
                    st['median'] if st['median'] is not None else "N/A",
                    st['std'] if st['std'] is not None else "N/A",
                    st['variance'] if st['variance'] is not None else "N/A",
                    st['min'] if st['min'] is not None else "N/A",
                    st['max'] if st['max'] is not None else "N/A",
                    st.get('q1', st.get('p25', "N/A")),
                    st.get('q3', st.get('p75', "N/A")),
                    st.get('iqr', "N/A"),
                    st['p10'] if st['p10'] is not None else "N/A",
                    st['p90'] if st['p90'] is not None else "N/A",
                    ref_low if ref_low is not None else "N/A",
                    ref_high if ref_high is not None else "N/A"
                ])

        # Automatically generate comprehensive box plots across all reference videos
        try:
            from src.calibration.biomechanical_statistics import BiomechanicalStatisticsAnalyzer
            analyzer = BiomechanicalStatisticsAnalyzer(reference_data, shot_type="cover_drive")
            box_plot_path = self.output_dir / "cover_drive_boxplots.png"
            analyzer.plot_all_metric_boxplots(save_path=box_plot_path)
        except Exception as e:
            print(f"[!] Warning: Could not generate box plot figure: {e}")

        return reference_data
