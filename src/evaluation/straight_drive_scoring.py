"""
AthleteRise - Straight Drive Scoring Engine
Dedicated evaluation authority for Straight Drive shots.
"""

from src.evaluation.base_scoring_engine import MultiShotScoringEngine

class StraightDriveScoringEngine(MultiShotScoringEngine):
    def __init__(self, reference_path="output/reference_calibration/straight_drive/straight_drive_reference.json", config_path="config.json"):
        super().__init__(shot_type="straight_drive", reference_path=reference_path, config_path=config_path)
