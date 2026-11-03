"""
Pull Shot Phase Detector
Specialized phase segmentation and contact moment estimation for Pull Shots.
"""

import numpy as np
from typing import Dict, List, Any
from src.phase_detection.base_phase_detector import BasePhaseDetector

class PullShotPhaseDetector(BasePhaseDetector):
    """
    Automatic Phase & Contact Moment Detector for Pull Shots.
    Phases:
      - Stance
      - Back-foot Transfer & Preparation
      - Horizontal Cross-Bat Swing
      - Contact Window
      - Leg-Side Follow-Through
    """
    def __init__(self, window_size: int = 5):
        super().__init__(shot_type="pull_shot", window_size=window_size)

    def detect_contact_frame(self, frame_metrics: List[Dict[str, Any]], total_frames: int) -> int:
        """
        Estimates contact frame based on peak shoulder rotational velocity
        and maximum horizontal arm extension through the contact zone.
        """
        if not frame_metrics:
            return int(total_frames // 2) if total_frames > 0 else 1

        valid_frames = [fm for fm in frame_metrics if fm.get('metrics') is not None]
        if len(valid_frames) < 3:
            return int(total_frames // 2) if total_frames > 0 else 1

        # Look for peak shoulder rotation in middle half
        rotations = []
        frames = []
        for fm in valid_frames:
            f_idx = fm['frame_idx']
            if 0.25 * total_frames <= f_idx <= 0.75 * total_frames:
                sh_rot = fm['metrics'].get('shoulder_rotation')
                if sh_rot is not None:
                    rotations.append(sh_rot)
                    frames.append(f_idx)

        if rotations:
            # Impact occurs near peak shoulder uncoiling
            max_idx = int(np.argmax(rotations))
            return int(frames[max_idx])

        return int(total_frames // 2) if total_frames > 0 else 1

    def segment_phases(self, frame_metrics: List[Dict[str, Any]], contact_frame: int, total_frames: int) -> Dict[str, List[int]]:
        prep_start = max(1, contact_frame - 18)
        swing_start = max(1, contact_frame - 8)
        follow_through_end = min(total_frames, contact_frame + 20)

        phases = {
            'stance': [1, max(1, prep_start - 1)],
            'back_foot_preparation': [prep_start, max(prep_start, swing_start - 1)],
            'horizontal_swing': [swing_start, max(swing_start, contact_frame - 1)],
            'contact_window': [max(1, contact_frame - 3), min(total_frames, contact_frame + 3)],
            'follow_through': [min(total_frames, contact_frame + 4), follow_through_end]
        }
        return phases
