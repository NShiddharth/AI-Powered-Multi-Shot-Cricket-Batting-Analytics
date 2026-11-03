"""
AthleteRise - Bat-Ball Contact Detection Package
"""

from src.contact_detection.bat_ball_detector import BatDetector, BallDetector
from src.contact_detection.contact_features import point_to_segment_distance, compute_trajectory_deflection
from src.contact_detection.contact_estimator import ContactEstimator, STATUS_CONTACT, STATUS_NO_CONTACT, STATUS_UNKNOWN
from src.contact_detection.annotation_manager import ContactAnnotationManager
