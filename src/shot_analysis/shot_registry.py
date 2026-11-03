"""
AthleteRise - Shot Registry
Provides an extensible registry pattern allowing new shot types to be added
without modifying core pipeline logic.
"""

from typing import Dict, Any, Type, Optional
from src.shot_analysis.shot_config import (
    SHOT_COVER_DRIVE,
    SHOT_STRAIGHT_DRIVE,
    SHOT_PULL_SHOT,
    DEFAULT_SHOT_CONFIG
)
from src.phase_detection.base_phase_detector import BasePhaseDetector
from src.phase_detection.cover_drive_phase_detector import CoverDrivePhaseDetector
from src.phase_detection.straight_drive_phase_detector import StraightDrivePhaseDetector
from src.phase_detection.pull_shot_phase_detector import PullShotPhaseDetector

class ShotRegistry:
    """
    Central registry for cricket batting shot definitions, detectors, and metadata.
    """
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, shot_type: str, name: str, phase_detector_cls: Type[BasePhaseDetector], reference_path: str, reference_file: str, description: str = ""):
        code = shot_type.lower().strip()
        cls._registry[code] = {
            'code': code,
            'name': name,
            'phase_detector_cls': phase_detector_cls,
            'reference_path': reference_path,
            'reference_file': reference_file,
            'description': description
        }

    @classmethod
    def get(cls, shot_type: str) -> Optional[Dict[str, Any]]:
        return cls._registry.get(shot_type.lower().strip())

    @classmethod
    def get_all(cls) -> Dict[str, Dict[str, Any]]:
        return dict(cls._registry)

    @classmethod
    def list_shots(cls) -> list:
        return list(cls._registry.keys())

# Register default 3 shots
ShotRegistry.register(
    shot_type=SHOT_COVER_DRIVE,
    name="Cover Drive",
    phase_detector_cls=CoverDrivePhaseDetector,
    reference_path="data/reference_videos/cover_drive",
    reference_file="output/reference_calibration/cover_drive/cover_drive_reference.json",
    description="Front-foot attacking stroke driven through extra-cover / off-side."
)

ShotRegistry.register(
    shot_type=SHOT_STRAIGHT_DRIVE,
    name="Straight Drive",
    phase_detector_cls=StraightDrivePhaseDetector,
    reference_path="data/reference_videos/straight_drive",
    reference_file="output/reference_calibration/straight_drive/straight_drive_reference.json",
    description="Front-foot drive executed straight back past the bowler."
)

ShotRegistry.register(
    shot_type=SHOT_PULL_SHOT,
    name="Pull Shot",
    phase_detector_cls=PullShotPhaseDetector,
    reference_path="data/reference_videos/pull_shot",
    reference_file="output/reference_calibration/pull_shot/pull_shot_reference.json",
    description="Cross-bat attacking shot played off the back foot through mid-wicket / square leg."
)
