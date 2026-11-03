"""
Phase Detector Factory
Dispatches shot-specific phase detectors dynamically by shot type.
"""

from typing import Optional
from src.phase_detection.base_phase_detector import BasePhaseDetector
from src.phase_detection.cover_drive_phase_detector import CoverDrivePhaseDetector
from src.phase_detection.straight_drive_phase_detector import StraightDrivePhaseDetector
from src.phase_detection.pull_shot_phase_detector import PullShotPhaseDetector

class PhaseDetectorFactory:
    """
    Factory creating appropriate phase detector instance based on shot class.
    """
    @staticmethod
    def create(shot_type: str = "cover_drive", window_size: int = 5) -> BasePhaseDetector:
        normalized = shot_type.lower().replace(" ", "_").strip()
        if normalized == "straight_drive":
            return StraightDrivePhaseDetector(window_size=window_size)
        elif normalized == "pull_shot":
            return PullShotPhaseDetector(window_size=window_size)
        else:
            # Default to Cover Drive detector (covers unknown and cover_drive)
            return CoverDrivePhaseDetector(window_size=window_size)
