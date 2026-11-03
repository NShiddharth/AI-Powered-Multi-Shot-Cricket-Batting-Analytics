"""
AthleteRise - Bat-Ball Contact Features & Spatial Geometry
Computes geometric proximity, trajectory deflection, and relative kinematics
between bat segment and cricket ball.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

def point_to_segment_distance(pt: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """
    Computes exact Euclidean distance from a point to a finite 2D line segment.
    pt: [x, y]
    seg_start: [x1, y1]
    seg_end: [x2, y2]
    """
    p = np.array(pt, dtype=np.float32)
    a = np.array(seg_start, dtype=np.float32)
    b = np.array(seg_end, dtype=np.float32)

    ab = b - a
    seg_len_sq = np.dot(ab, ab)
    if seg_len_sq < 1e-6:
        return float(np.linalg.norm(p - a))

    # Projection factor t clamped to [0, 1]
    t = max(0.0, min(1.0, np.dot(p - a, ab) / seg_len_sq))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))

def compute_trajectory_angle(points: List[Tuple[float, float]]) -> Optional[float]:
    """
    Computes heading angle in degrees (0 to 360) from a series of 2D points.
    """
    if len(points) < 2:
        return None
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return None
    deg = np.degrees(np.arctan2(dy, dx))
    return float(deg % 360.0)

def compute_trajectory_deflection(incoming_pts: List[Tuple[float, float]], outgoing_pts: List[Tuple[float, float]]) -> float:
    """
    Computes angle change in degrees between incoming and outgoing trajectories.
    """
    a_in = compute_trajectory_angle(incoming_pts)
    a_out = compute_trajectory_angle(outgoing_pts)
    if a_in is None or a_out is None:
        return 0.0
    diff = abs(a_out - a_in)
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)
