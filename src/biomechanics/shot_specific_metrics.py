"""
AthleteRise - Shot-Specific Biomechanical Metrics
Calculates specialized biomechanical metrics for:
- Cover Drive (off-side bat vector, cover foot alignment, front arm extension)
- Straight Drive (vertical presentation, straight down-pitch line, head-knee verticality)
- Pull Shot (rotational trunk torque, horizontal plane index, back-foot weight dominance)
"""

import numpy as np
from typing import Dict, Any, Optional

def compute_shot_specific_metrics(shot_type: str, contact_metrics: Dict[str, Any], raw_landmarks: Optional[Dict[str, Any]] = None, handedness: str = "right") -> Dict[str, Any]:
    """
    Computes specialized metric indicators depending on the detected shot type.
    Extracts geometric indices from instantaneous contact window measurements.
    """
    shot_type = (shot_type or "cover_drive").lower()
    specifics = {}

    if shot_type == "cover_drive":
        # Cover Drive:
        # 1. Front Foot Cover Alignment (°): optimal between 140° and 175°
        front_foot_dir = contact_metrics.get('front_foot_direction')
        specifics['cover_direction_alignment'] = front_foot_dir

        # 2. Bat Extension Index: Front elbow angle relative to full reach
        elbow = contact_metrics.get('front_elbow_angle')
        specifics['bat_extension_index'] = float(np.clip(elbow / 180.0, 0.0, 1.0)) if elbow is not None else None

        # 3. Off-Side Flow Vector: Combined stride and shoulder uncoiling toward extra cover
        sh_rot = contact_metrics.get('shoulder_rotation')
        stride = contact_metrics.get('stride_length')
        if sh_rot is not None and stride is not None:
            specifics['offside_flow_vector'] = round(float(sh_rot * 0.5 + stride * 100 * 0.5), 2)
        else:
            specifics['offside_flow_vector'] = None

    elif shot_type == "straight_drive":
        # Straight Drive:
        # 1. Straight Line Presentation: Foot directed straight down pitch (~85°-115°)
        foot_dir = contact_metrics.get('front_foot_direction')
        specifics['straight_bat_presentation'] = foot_dir

        # 2. Vertical Head-Over-Ball Index: Lower horizontal offset indicates ideal verticality
        hk_align = contact_metrics.get('head_knee_alignment')
        if hk_align is not None:
            # 1.0 = perfect vertical alignment, decreasing as offset grows
            specifics['head_over_ball_verticality'] = round(float(max(0.0, 1.0 - hk_align)), 3)
        else:
            specifics['head_over_ball_verticality'] = None

        # 3. Leading Elbow Height: High elbow drives straight presentation
        elbow = contact_metrics.get('front_elbow_angle')
        specifics['leading_elbow_elevation'] = elbow

    elif shot_type == "pull_shot":
        # Pull Shot:
        # 1. Trunk Rotational Uncoiling (°): High rotation uncoiling to leg-side
        sh_rot = contact_metrics.get('shoulder_rotation')
        specifics['rotary_trunk_uncoiling'] = sh_rot

        # 2. Horizontal Plane Index: Measures horizontal cross-bat execution vs vertical drive
        spine = contact_metrics.get('spine_lean')
        elbow = contact_metrics.get('front_elbow_angle')
        if spine is not None and elbow is not None:
            # Cross-bat shots exhibit lateral spine tilt combined with horizontal arm elevation
            plane_index = float(np.clip((spine / 30.0) * 50.0 + (elbow / 150.0) * 50.0, 0.0, 100.0))
            specifics['horizontal_plane_index'] = round(plane_index, 1)
        else:
            specifics['horizontal_plane_index'] = None

        # 3. Weight Transfer / Stride Compactness: Pull shot requires compact stride off back foot
        stride = contact_metrics.get('stride_length')
        if stride is not None:
            # Normalized back-foot compactness (lower forward stride is more compact)
            specifics['rear_foot_pivot_compactness'] = round(float(max(0.0, 1.0 - stride)), 3)
        else:
            specifics['rear_foot_pivot_compactness'] = None

    return specifics
