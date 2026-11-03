"""
AthleteRise - Pull Shot Calibration Module
Specialized calibration engine for Pull Shot reference videos.
"""

from src.calibration.base_calibration_engine import ShotCalibrationEngine

class PullShotCalibrationEngine(ShotCalibrationEngine):
    def __init__(self, reference_dir="data/reference_videos/pull_shot", output_dir="output/reference_calibration/pull_shot", stance_override=None):
        super().__init__(shot_type="pull_shot", reference_dir=reference_dir, output_dir=output_dir, stance_override=stance_override)
