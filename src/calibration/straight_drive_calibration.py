"""
AthleteRise - Straight Drive Calibration Module
Specialized calibration engine for Straight Drive reference videos.
"""

from src.calibration.base_calibration_engine import ShotCalibrationEngine

class StraightDriveCalibrationEngine(ShotCalibrationEngine):
    def __init__(self, reference_dir="data/reference_videos/straight_drive", output_dir="output/reference_calibration/straight_drive", stance_override=None):
        super().__init__(shot_type="straight_drive", reference_dir=reference_dir, output_dir=output_dir, stance_override=stance_override)
