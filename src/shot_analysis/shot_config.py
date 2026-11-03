"""
AthleteRise - Multi-Shot Configuration & Registry Metadata
Centralizes shot identifiers, directory paths, reference JSON destinations,
and contact detection parameters.
"""

import os
from pathlib import Path

# Supported Shot Identifiers
SHOT_COVER_DRIVE = "cover_drive"
SHOT_STRAIGHT_DRIVE = "straight_drive"
SHOT_PULL_SHOT = "pull_shot"
SHOT_UNKNOWN = "unknown"

SUPPORTED_SHOTS = [
    SHOT_COVER_DRIVE,
    SHOT_STRAIGHT_DRIVE,
    SHOT_PULL_SHOT
]

SHOT_DISPLAY_NAMES = {
    SHOT_COVER_DRIVE: "Cover Drive",
    SHOT_STRAIGHT_DRIVE: "Straight Drive",
    SHOT_PULL_SHOT: "Pull Shot",
    SHOT_UNKNOWN: "Unknown Shot"
}

DEFAULT_SHOT_CONFIG = {
    "shots": {
        SHOT_COVER_DRIVE: {
            "name": "Cover Drive",
            "code": SHOT_COVER_DRIVE,
            "reference_path": "data/reference_videos/cover_drive",
            "reference_file": "output/reference_calibration/cover_drive/cover_drive_reference.json",
            "description": "Front-foot attacking stroke driven through extra-cover / off-side."
        },
        SHOT_STRAIGHT_DRIVE: {
            "name": "Straight Drive",
            "code": SHOT_STRAIGHT_DRIVE,
            "reference_path": "data/reference_videos/straight_drive",
            "reference_file": "output/reference_calibration/straight_drive/straight_drive_reference.json",
            "description": "Front-foot drive executed straight back past the bowler with vertical bat presentation."
        },
        SHOT_PULL_SHOT: {
            "name": "Pull Shot",
            "code": SHOT_PULL_SHOT,
            "reference_path": "data/reference_videos/pull_shot",
            "reference_file": "output/reference_calibration/pull_shot/pull_shot_reference.json",
            "description": "Cross-bat attacking shot played off the back foot through mid-wicket / square leg."
        }
    },
    "classifier": {
        "confidence_threshold": 0.50,
        "margin_threshold": 0.12,
        "feature_count": 12
    },
    "contact_detection": {
        "enabled": True,
        "unknown_threshold": 0.45,
        "spatial_proximity_threshold": 90.0,
        "trajectory_angle_threshold": 25.0
    }
}

def get_shot_reference_path(shot_type: str) -> str:
    """Returns directory path containing reference videos for a shot type."""
    cfg = DEFAULT_SHOT_CONFIG["shots"].get(shot_type.lower())
    if cfg:
        return cfg["reference_path"]
    return f"data/reference_videos/{shot_type.lower()}"

def get_shot_reference_json(shot_type: str) -> str:
    """Returns file path of the calibrated reference JSON for a shot type."""
    cfg = DEFAULT_SHOT_CONFIG["shots"].get(shot_type.lower())
    if cfg:
        return cfg["reference_file"]
    return f"output/reference_calibration/{shot_type.lower()}/{shot_type.lower()}_reference.json"
