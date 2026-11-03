"""
Cover Drive Phase Detector
Specialized phase segmentation and contact estimation for Cover Drives.
"""

import numpy as np
from typing import Dict, List, Any
from src.phase_detection.base_phase_detector import BasePhaseDetector

class CoverDrivePhaseDetector(BasePhaseDetector):
    """
    Automatic Phase & Contact Moment Detector for Cover Drive shots.
    Identifies:
      - Stance
      - Downswing
      - Contact / Impact Frame
      - Follow-through
    """
    def __init__(self, window_size: int = 5):
        super().__init__(shot_type="cover_drive", window_size=window_size)

    def detect_contact_frame(self, frame_metrics: List[Dict[str, Any]], total_frames: int) -> int:
        """
        Detects contact frame index (1-based) based on biomechanical cues:
        1. Downswing impact inflection (minimum front elbow angle followed by follow-through extension)
        2. Peak stride length / head forward progression
        """
        if not frame_metrics:
            return int(total_frames // 2) if total_frames > 0 else 1

        valid_frames = [fm for fm in frame_metrics if fm.get('metrics') is not None]
        if not valid_frames:
            return int(total_frames // 2) if total_frames > 0 else 1

        mid_frames = [
            (fm['frame_idx'], fm['metrics']['front_elbow_angle'])
            for fm in valid_frames
            if fm['metrics'].get('front_elbow_angle') is not None and 0.20 * total_frames <= fm['frame_idx'] <= 0.85 * total_frames
        ]
        if not mid_frames:
            mid_frames = [
                (fm['frame_idx'], fm['metrics']['front_elbow_angle'])
                for fm in valid_frames
                if fm['metrics'].get('front_elbow_angle') is not None
            ]

        if not mid_frames:
            return int(total_frames // 2) if total_frames > 0 else 1

        # Stroke impact inflection:
        # A true downswing impact is followed by arm extension into the follow-through.
        candidate_inflections = []
        for i, (f_idx, angle) in enumerate(mid_frames):
            subsequent = [a for f, a in mid_frames[i+1:i+12]]
            if subsequent:
                max_extension = max(subsequent)
                rise = max_extension - angle
                if rise >= 25.0:
                    candidate_inflections.append((f_idx, angle, rise))

        if candidate_inflections:
            # Rank candidates: prioritize genuine downswing inflection with strong follow-through rise
            candidate_inflections.sort(key=lambda x: (x[1] - x[2] * 0.35))
            return int(candidate_inflections[0][0])

        min_frame = min(mid_frames, key=lambda x: x[1])[0]
        return int(min_frame)

    def segment_phases(self, frame_metrics: List[Dict[str, Any]], contact_frame: int, total_frames: int) -> Dict[str, List[int]]:
        """
        Segments complete shot into 4 core phases relative to detected contact frame.
        """
        downswing_start = max(1, contact_frame - 15)
        follow_through_end = min(total_frames, contact_frame + 20)

        phases = {
            'stance': [1, max(1, downswing_start - 1)],
            'downswing': [downswing_start, max(downswing_start, contact_frame - 1)],
            'contact_window': [max(1, contact_frame - 3), min(total_frames, contact_frame + 3)],
            'follow_through': [min(total_frames, contact_frame + 4), follow_through_end]
        }
        return phases
