"""
AthleteRise - Pull Shot Scoring Engine
Dedicated evaluation authority for Pull Shot strokes.
"""

from src.evaluation.base_scoring_engine import MultiShotScoringEngine

class PullShotScoringEngine(MultiShotScoringEngine):
    def __init__(self, reference_path="output/reference_calibration/pull_shot/pull_shot_reference.json", config_path="config.json"):
        super().__init__(shot_type="pull_shot", reference_path=reference_path, config_path=config_path)
