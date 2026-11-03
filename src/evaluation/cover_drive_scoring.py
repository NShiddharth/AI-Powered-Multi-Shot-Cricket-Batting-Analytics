"""
AthleteRise - Cover Drive Scoring Engine
Dedicated evaluation authority for Cover Drive shots.
"""

from src.evaluation.base_scoring_engine import MultiShotScoringEngine

class CoverDriveScoringEngine(MultiShotScoringEngine):
    def __init__(self, reference_path="output/reference_calibration/cover_drive/cover_drive_reference.json", config_path="config.json"):
        super().__init__(shot_type="cover_drive", reference_path=reference_path, config_path=config_path)
