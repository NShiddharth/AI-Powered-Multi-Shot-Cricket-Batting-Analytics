"""
Straight Drive Phase Detector
Specialized phase segmentation and contact moment estimation for Straight Drives.
"""

import numpy as np
from typing import Dict, List, Any
from src.phase_detection.base_phase_detector import BasePhaseDetector

class StraightDrivePhaseDetector(BasePhaseDetector):
    """
    Automatic Phase & Contact Moment Detector for Straight Drive shots.
    Phases:
      - Stance
      - Forward Stride & Downswing
      - Contact Window (head over ball, straight bat impact)
      - High Vertical Follow-through
    """
    def __init__(self, window_size: int = 5):
        super().__init__(shot_type="straight_drive", window_size=window_size)

    def detect_contact_frame(self, frame_metrics: List[Dict[str, Any]], total_frames: int) -> int:
        """
        Estimates contact frame based on peak forward head alignment with front knee
        and maximum wrist downward progression.
        """
        if not frame_metrics:
            return int(total_frames // 2) if total_frames > 0 else 1

        valid_frames = [fm for fm in frame_metrics if fm.get('metrics') is not None]
        if not valid_frames:
            return int(total_frames // 2) if total_frames > 0 else 1

        # Look for minimum head-knee horizontal offset and minimum front elbow angle in middle third
        candidates = []
        for fm in valid_frames:
            m = fm['metrics']
            f_idx = fm['frame_idx']
            # Only consider frames in the middle 60% of the video
            if 0.20 * total_frames <= f_idx <= 0.80 * total_frames:
                align = m.get('head_knee_alignment') or 1.0
                elbow = m.get('front_elbow_angle') or 90.0
                score = align * 0.6 + (abs(elbow - 130.0) / 180.0) * 0.4
                candidates.append((score, f_idx))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return int(candidates[0][1])

        return int(total_frames // 2) if total_frames > 0 else 1

    def segment_phases(self, frame_metrics: List[Dict[str, Any]], contact_frame: int, total_frames: int) -> Dict[str, List[int]]:
        downswing_start = max(1, contact_frame - 14)
        follow_through_end = min(total_frames, contact_frame + 22)

        phases = {
            'stance': [1, max(1, downswing_start - 1)],
            'stride_and_downswing': [downswing_start, max(downswing_start, contact_frame - 1)],
            'contact_window': [max(1, contact_frame - 3), min(total_frames, contact_frame + 3)],
            'follow_through': [min(total_frames, contact_frame + 4), follow_through_end]
        }
        return phases
