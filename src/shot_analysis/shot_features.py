"""
AthleteRise - Shot Feature Extractor
Extracts spatio-temporal kinematic and biomechanical feature vectors
for data-driven multi-shot classification.
Provides full feature availability, quality tracking, and body-scale normalization.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

FEATURE_NAMES = [
    'front_foot_direction',
    'stride_length',
    'shoulder_rotation',
    'hip_rotation',
    'front_elbow_angle',
    'spine_lean',
    'head_knee_alignment',
    'rotary_shoulder_diff',
    'wrist_horizontal_travel',
    'wrist_vertical_travel',
    'follow_through_angle',
    'head_stability'
]

class ShotFeatureExtractor:
    """
    Extracts high-dimensional spatio-temporal features from processed frame metrics
    and pose landmarks across video frames.
    Tracks feature availability, quality, and physiological consistency.
    """
    def __init__(self, handedness: str = "right"):
        self.handedness = handedness

    def extract_features(self, frame_metrics_list: List[Dict[str, Any]], contact_frame: int, total_frames: int) -> Dict[str, float]:
        """
        Extracts the 12-dimensional kinematic feature dictionary from frame metrics.
        Handles missing or occluded values robustly using phase medians and temporal tracking.
        """
        details = self.extract_feature_details(frame_metrics_list, contact_frame, total_frames)
        return {k: round(v['value'], 4) if v['value'] is not None else 0.0 for k, v in details.items()}

    def extract_feature_details(self, frame_metrics_list: List[Dict[str, Any]], contact_frame: int, total_frames: int) -> Dict[str, Dict[str, Any]]:
        """
        Extracts feature values with explicit availability, quality, and explanation reason.
        Conforms strictly to Section 11 & Section 28 of the project architecture:
        {
            "feature_name": {
                "value": float or None,
                "available": bool,
                "quality": float (0.0 to 1.0),
                "reason": str
            }
        }
        """
        if not frame_metrics_list:
            return {
                k: {
                    'value': None,
                    'available': False,
                    'quality': 0.0,
                    'reason': 'No video frame metrics provided.'
                } for k in FEATURE_NAMES
            }

        valid_fms = [fm for fm in frame_metrics_list if fm.get('metrics') is not None]
        if not valid_fms:
            return {
                k: {
                    'value': None,
                    'available': False,
                    'quality': 0.0,
                    'reason': 'Pose estimation failed across all frames.'
                } for k in FEATURE_NAMES
            }

        # 1. Stroke Event Windows
        # Contact Window: +/- 3 frames around estimated impact
        c_min = max(1, contact_frame - 3)
        c_max = min(total_frames, contact_frame + 3)
        cw = [fm['metrics'] for fm in valid_fms if c_min <= fm['frame_idx'] <= c_max]
        if not cw:
            cw = [fm['metrics'] for fm in valid_fms]
        cw_count = len(cw)

        def eval_metric_window(key: str, default_val: float, valid_min: float = -999.0, valid_max: float = 999.0) -> Tuple[Optional[float], bool, float, str]:
            vals = [m[key] for m in cw if key in m and m[key] is not None and valid_min <= m[key] <= valid_max]
            if len(vals) >= max(1, int(cw_count * 0.35)):
                val = float(np.median(vals))
                quality = min(1.0, len(vals) / max(1, cw_count))
                return val, True, round(quality, 2), f"Extracted from {len(vals)}/{cw_count} valid frames in contact window [{c_min}-{c_max}]."
            elif vals:
                val = float(np.median(vals))
                quality = round(len(vals) / max(1, cw_count), 2)
                return val, True, quality, f"Extracted from partial contact window ({len(vals)}/{cw_count} frames)."
            else:
                return default_val, False, 0.0, f"Landmarks missing or occluded in >70% of contact window [{c_min}-{c_max}]."

        # Front foot direction
        ffd_val, ffd_avail, ffd_qual, ffd_reas = eval_metric_window('front_foot_direction', 160.0, 45.0, 180.0)

        # Stride length: Evaluated across complete delivery movement rather than single occluded instant
        movement_frames = [fm['metrics'] for fm in valid_fms if fm['frame_idx'] <= min(total_frames, contact_frame + 4) and fm.get('metrics')]
        stride_candidates = [
            m['stride_length'] for m in movement_frames
            if m.get('stride_length') is not None and 0.05 <= m['stride_length'] <= 1.20
        ]
        if len(stride_candidates) >= 3:
            # Stride extension peaks as front foot plants during delivery
            stride_val = float(np.percentile(stride_candidates, 85))
            stride_avail = True
            stride_qual = min(1.0, len(stride_candidates) / max(1, len(movement_frames)))
            stride_reas = f"Calculated from complete movement delivery phase ({len(stride_candidates)} valid tracking frames)."
        else:
            stride_val, stride_avail, stride_qual, stride_reas = eval_metric_window('stride_length', 0.22, 0.02, 1.20)

        # Shoulder rotation (consistent uncoiling)
        sh_val, sh_avail, sh_qual, sh_reas = eval_metric_window('shoulder_rotation', 135.0, 10.0, 180.0)

        # Hip rotation (consistent uncoiling)
        hip_val, hip_avail, hip_qual, hip_reas = eval_metric_window('hip_rotation', 135.0, 10.0, 180.0)

        # Front elbow angle (anatomical limits 25 to 180)
        elbow_val, elbow_avail, elbow_qual, elbow_reas = eval_metric_window('front_elbow_angle', 100.0, 25.0, 180.0)

        # Torso scale calculation for camera distance scale-invariance
        torso_scales = []
        for fm in valid_fms:
            lm = fm.get('landmarks')
            if lm and 'left_shoulder' in lm and 'left_hip' in lm and 'right_shoulder' in lm and 'right_hip' in lm:
                sm = [(lm['left_shoulder'][0] + lm['right_shoulder'][0]) / 2.0, (lm['left_shoulder'][1] + lm['right_shoulder'][1]) / 2.0]
                hm = [(lm['left_hip'][0] + lm['right_hip'][0]) / 2.0, (lm['left_hip'][1] + lm['right_hip'][1]) / 2.0]
                torso_scales.append(float(np.linalg.norm(np.array(sm) - np.array(hm))))
        torso_scale = float(np.median(torso_scales)) if torso_scales else 0.20
        torso_scale = max(0.05, torso_scale)

        # Spine lean: Evaluated across delivery downswing and contact window
        # (contact_frame - 8 to contact_frame + 2) to capture true forward bend
        ds_c_frames = [fm['metrics'] for fm in valid_fms if max(1, contact_frame - 8) <= fm['frame_idx'] <= min(total_frames, contact_frame + 2) and fm.get('metrics')]
        spine_candidates = [m['spine_lean'] for m in ds_c_frames if m.get('spine_lean') is not None and 0.0 <= m['spine_lean'] <= 60.0]
        if spine_candidates:
            spine_val = float(np.percentile(spine_candidates, 75))
            spine_avail = True
            spine_qual = min(1.0, len(spine_candidates) / max(1, len(ds_c_frames)))
            spine_reas = f"Extracted peak delivery lean over {len(spine_candidates)} frames in delivery window."
        else:
            spine_val, spine_avail, spine_qual, spine_reas = eval_metric_window('spine_lean', 10.0, 0.0, 60.0)

        # Head knee alignment
        head_knee_val, head_knee_avail, head_knee_qual, head_knee_reas = eval_metric_window('head_knee_alignment', 0.25, 0.0, 2.0)

        # 2. Phase-dependent temporal dynamics
        # Head stability during downswing/pre-impact window
        ds_frames = [fm['metrics'] for fm in valid_fms if fm['frame_idx'] <= contact_frame and fm.get('metrics')]
        if len(ds_frames) >= 2:
            n_xs = [m['nose_x_norm'] for m in ds_frames if m.get('nose_x_norm') is not None]
            n_ys = [m['nose_y_norm'] for m in ds_frames if m.get('nose_y_norm') is not None]
            if len(n_xs) >= 2:
                head_stab_val = float(np.sqrt(np.var(n_xs) + np.var(n_ys)))
                head_stab_avail = True
                head_stab_qual = min(1.0, len(n_xs) / max(1, len(ds_frames)))
                head_stab_reas = f"Computed spatial variance over {len(n_xs)} downswing frames."
            else:
                head_stab_val = 20.0
                head_stab_avail = False
                head_stab_qual = 0.2
                head_stab_reas = "Insufficient nose tracking points during downswing."
        else:
            head_stab_val = 20.0
            head_stab_avail = False
            head_stab_qual = 0.0
            head_stab_reas = "Downswing phase too brief for head stability calculation."

        # Follow-through angle from post-contact frames
        ft_frames = [fm['metrics'] for fm in valid_fms if fm['frame_idx'] > contact_frame and fm.get('metrics')]
        if ft_frames:
            ft_angles = [m['front_elbow_angle'] for m in ft_frames if m.get('front_elbow_angle') is not None and 25.0 <= m['front_elbow_angle'] <= 180.0]
            if ft_angles:
                ft_angle_val = float(np.mean(ft_angles))
                ft_avail = True
                ft_qual = min(1.0, len(ft_angles) / max(1, len(ft_frames)))
                ft_reas = f"Averaged over {len(ft_angles)} post-contact follow-through frames."
            else:
                ft_angle_val = elbow_val
                ft_avail = False
                ft_qual = 0.2
                ft_reas = "Front elbow occluded during post-contact follow-through."
        else:
            ft_angle_val = elbow_val
            ft_avail = False
            ft_qual = 0.0
            ft_reas = "No post-contact follow-through frames present in video."

        # Rotary shoulder difference between early preparation and post-impact
        early_frames = [fm['metrics'] for fm in valid_fms if fm['frame_idx'] <= max(2, int(total_frames * 0.25))]
        late_frames = [fm['metrics'] for fm in valid_fms if fm['frame_idx'] >= min(total_frames - 2, int(total_frames * 0.70))]

        early_sh_vals = [m['shoulder_rotation'] for m in early_frames if m.get('shoulder_rotation') is not None]
        late_sh_vals = [m['shoulder_rotation'] for m in late_frames if m.get('shoulder_rotation') is not None]

        if early_sh_vals and late_sh_vals:
            early_sh = float(np.median(early_sh_vals))
            late_sh = float(np.median(late_sh_vals))
            rotary_diff_val = abs(late_sh - early_sh)
            rot_avail = True
            rot_qual = min(1.0, (len(early_sh_vals) + len(late_sh_vals)) / max(2, len(early_frames) + len(late_frames)))
            rot_reas = f"Computed difference between preparation ({len(early_sh_vals)} frames) and follow-through ({len(late_sh_vals)} frames)."
        else:
            rotary_diff_val = 50.0
            rot_avail = False
            rot_qual = 0.1
            rot_reas = "Insufficient shoulder tracking in preparation or follow-through phases."

        # 3. Wrist displacement dynamics (normalized by torso scale)
        wrist_key = 'left_wrist' if self.handedness == 'right' else 'right_wrist'
        wrist_xs = []
        wrist_ys = []
        for fm in valid_fms:
            lm = fm.get('landmarks')
            if lm and wrist_key in lm:
                wrist_xs.append(lm[wrist_key][0])
                wrist_ys.append(lm[wrist_key][1])

        if len(wrist_xs) >= 2:
            raw_h = float(abs(max(wrist_xs) - min(wrist_xs)))
            raw_v = float(abs(max(wrist_ys) - min(wrist_ys)))
            wrist_h_val = float(raw_h / torso_scale)
            wrist_v_val = float(raw_v / torso_scale)
            wrist_avail = True
            wrist_qual = min(1.0, len(wrist_xs) / max(1, total_frames))
            wrist_reas = f"Tracked wrist coordinates normalized by torso scale ({round(torso_scale, 3)}) across {len(wrist_xs)}/{total_frames} frames."
        else:
            wrist_h_val = 2.5
            wrist_v_val = 5.5
            wrist_avail = False
            wrist_qual = 0.0
            wrist_reas = "Wrist landmarks missing in >80% of video frames."

        return {
            'front_foot_direction': {
                'value': round(ffd_val, 2) if ffd_val is not None else None,
                'available': ffd_avail,
                'quality': round(ffd_qual, 2),
                'reason': ffd_reas
            },
            'stride_length': {
                'value': round(stride_val, 4) if stride_val is not None else None,
                'available': stride_avail,
                'quality': round(stride_qual, 2),
                'reason': stride_reas
            },
            'shoulder_rotation': {
                'value': round(sh_val, 2) if sh_val is not None else None,
                'available': sh_avail,
                'quality': round(sh_qual, 2),
                'reason': sh_reas
            },
            'hip_rotation': {
                'value': round(hip_val, 2) if hip_val is not None else None,
                'available': hip_avail,
                'quality': round(hip_qual, 2),
                'reason': hip_reas
            },
            'front_elbow_angle': {
                'value': round(elbow_val, 2) if elbow_val is not None else None,
                'available': elbow_avail,
                'quality': round(elbow_qual, 2),
                'reason': elbow_reas
            },
            'spine_lean': {
                'value': round(spine_val, 2) if spine_val is not None else None,
                'available': spine_avail,
                'quality': round(spine_qual, 2),
                'reason': spine_reas
            },
            'head_knee_alignment': {
                'value': round(head_knee_val, 4) if head_knee_val is not None else None,
                'available': head_knee_avail,
                'quality': round(head_knee_qual, 2),
                'reason': head_knee_reas
            },
            'rotary_shoulder_diff': {
                'value': round(rotary_diff_val, 2) if rotary_diff_val is not None else None,
                'available': rot_avail,
                'quality': round(rot_qual, 2),
                'reason': rot_reas
            },
            'wrist_horizontal_travel': {
                'value': round(wrist_h_val, 4) if wrist_h_val is not None else None,
                'available': wrist_avail,
                'quality': round(wrist_qual, 2),
                'reason': wrist_reas
            },
            'wrist_vertical_travel': {
                'value': round(wrist_v_val, 4) if wrist_v_val is not None else None,
                'available': wrist_avail,
                'quality': round(wrist_qual, 2),
                'reason': wrist_reas
            },
            'follow_through_angle': {
                'value': round(ft_angle_val, 2) if ft_angle_val is not None else None,
                'available': ft_avail,
                'quality': round(ft_qual, 2),
                'reason': ft_reas
            },
            'head_stability': {
                'value': round(head_stab_val, 4) if head_stab_val is not None else None,
                'available': head_stab_avail,
                'quality': round(head_stab_qual, 2),
                'reason': head_stab_reas
            }
        }

    def to_feature_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Converts feature dictionary to standardized NumPy array."""
        return np.array([feature_dict.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)
