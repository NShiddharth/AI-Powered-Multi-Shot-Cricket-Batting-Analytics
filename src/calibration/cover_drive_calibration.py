"""
AthleteRise - Cover Drive Calibration Module
Specialized calibration engine for Cover Drive reference videos.
"""

from src.calibration.base_calibration_engine import ShotCalibrationEngine

class CoverDriveCalibrationEngine(ShotCalibrationEngine):
    def __init__(self, reference_dir="data/reference_videos/cover_drive", output_dir="output/reference_calibration/cover_drive", stance_override=None):
        super().__init__(shot_type="cover_drive", reference_dir=reference_dir, output_dir=output_dir, stance_override=stance_override)
