"""
Calibration Factory
Instantiates calibration engines by shot type.
"""

from typing import Optional
from src.calibration.base_calibration_engine import ShotCalibrationEngine
from src.calibration.cover_drive_calibration import CoverDriveCalibrationEngine
from src.calibration.straight_drive_calibration import StraightDriveCalibrationEngine
from src.calibration.pull_shot_calibration import PullShotCalibrationEngine

class CalibrationFactory:
    """
    Factory creating appropriate calibration engine instance for a shot class.
    """
    @staticmethod
    def create(shot_type: str = "cover_drive", reference_dir: Optional[str] = None, output_dir: Optional[str] = None, stance_override: Optional[str] = None) -> ShotCalibrationEngine:
        normalized = shot_type.lower().replace(" ", "_").strip()
        if normalized == "straight_drive":
            return StraightDriveCalibrationEngine(reference_dir=reference_dir, output_dir=output_dir, stance_override=stance_override)
        elif normalized == "pull_shot":
            return PullShotCalibrationEngine(reference_dir=reference_dir, output_dir=output_dir, stance_override=stance_override)
        else:
            return CoverDriveCalibrationEngine(reference_dir=reference_dir, output_dir=output_dir, stance_override=stance_override)
