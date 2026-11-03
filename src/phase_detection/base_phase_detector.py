"""
Base Phase Detector Interface for Cricket Batting Analysis.
Provides standard contracts for detecting impact moments and shot phase segmentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BasePhaseDetector(ABC):
    """
    Abstract Base Class for shot-specific phase and contact detection.
    """
    def __init__(self, shot_type: str = "generic", window_size: int = 5):
        self.shot_type = shot_type
        self.window_size = window_size

    @abstractmethod
    def detect_contact_frame(self, frame_metrics: List[Dict[str, Any]], total_frames: int) -> int:
        """
        Estimates the biomechanical contact/impact frame index (1-based).
        """
        pass

    @abstractmethod
    def segment_phases(self, frame_metrics: List[Dict[str, Any]], contact_frame: int, total_frames: int) -> Dict[str, List[int]]:
        """
        Segments the shot into distinct temporal phases:
        e.g. stance, preparation/stride, downswing/swing, contact_window, follow_through.
        Returns dictionary mapping phase name to [start_frame, end_frame] (1-based, inclusive).
        """
        pass
