"""
Backward-compatible Phase Detector Module for AthleteRise.
Preserves CoverDrivePhaseDetector interface and exports PhaseDetectorFactory.
"""

from src.phase_detection.base_phase_detector import BasePhaseDetector
from src.phase_detection.cover_drive_phase_detector import CoverDrivePhaseDetector
from src.phase_detection.straight_drive_phase_detector import StraightDrivePhaseDetector
from src.phase_detection.pull_shot_phase_detector import PullShotPhaseDetector
from src.phase_detection.phase_factory import PhaseDetectorFactory

__all__ = [
    'BasePhaseDetector',
    'CoverDrivePhaseDetector',
    'StraightDrivePhaseDetector',
    'PullShotPhaseDetector',
    'PhaseDetectorFactory'
]
