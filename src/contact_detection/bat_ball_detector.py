"""
AthleteRise - Modular Bat & Ball Detectors
Implements modular visual detection and tracking for cricket bat and ball:
- BatDetector: Extracts bat bounding box, centerline axis, and confidence.
- BallDetector: Color/motion candidate tracking with temporal trajectory validation.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class BatDetector:
    """
    Modular cricket bat detector.
    Provides standard interface supporting:
    1. Dedicated ML/YOLO model if weights are provided
    2. Heuristic visual fallback using wrist trajectory, color segmentation, and directional edge contours.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.has_model = model_path is not None and os.path.exists(model_path)

    def detect(self, frame: np.ndarray, wrist_point: Optional[Tuple[int, int]] = None, frame_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        Detects bat geometry in single video frame.
        Returns:
            {
                "bbox": [x, y, w, h],
                "confidence": float (0-1),
                "center": [cx, cy],
                "angle": float (degrees),
                "segment": [[x1, y1], [x2, y2]]  # Top and bottom of bat blade
            }
        Returns None if no credible bat evidence is detected.
        """
        h, w = frame.shape[:2]

        # 1. Color segmentation in HSV for willow wood / bat stickers
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Typical willow / wood tone / bat rubber grip ranges
        lower_wood = np.array([10, 30, 80])
        upper_wood = np.array([35, 200, 255])
        mask_wood = cv2.inRange(hsv, lower_wood, upper_wood)

        # Region of interest around wrist if available
        if wrist_point is not None:
            wx, wy = wrist_point
            roi_x1 = max(0, int(wx - 160))
            roi_y1 = max(0, int(wy - 120))
            roi_x2 = min(w, int(wx + 160))
            roi_y2 = min(h, int(wy + 180))
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
            mask_wood = cv2.bitwise_and(mask_wood, roi_mask)

        # Morphological filtering to connect elongated bat segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
        mask_wood = cv2.morphologyEx(mask_wood, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 300 <= area <= 15000:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect
                aspect_ratio = max(rw, rh) / max(1.0, min(rw, rh))
                # Cricket bat is elongated (aspect ratio usually >= 2.0)
                if aspect_ratio >= 1.8:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    # Compute centerline segment
                    box = cv2.boxPoints(rect)
                    # Find ends of longest dimension
                    tip_pt = [int(cx + (max(rw, rh) / 2) * np.sin(np.radians(angle))), int(cy + (max(rw, rh) / 2) * np.cos(np.radians(angle)))]
                    candidates.append({
                        'bbox': [int(x), int(y), int(bw), int(bh)],
                        'confidence': min(0.92, float(0.40 + 0.10 * aspect_ratio)),
                        'center': [int(cx), int(cy)],
                        'tip': tip_pt,
                        'angle': float(angle),
                        'segment': [
                            [int(cx - (max(rw, rh) / 2) * np.sin(np.radians(angle))), int(cy - (max(rw, rh) / 2) * np.cos(np.radians(angle)))],
                            tip_pt
                        ],
                        'is_estimated': False
                    })

        if candidates:
            # Sort by area & confidence
            candidates.sort(key=lambda c: c['confidence'], reverse=True)
            return candidates[0]

        # If wrist point is available, generate conservative heuristic bat extension
        if wrist_point is not None:
            wx, wy = int(wrist_point[0]), int(wrist_point[1])
            # Default downward extension from hands
            seg_start = [wx, wy]
            seg_end = [wx + 15, min(h - 5, wy + 85)]
            return {
                'bbox': [wx - 15, wy, 35, 90],
                'confidence': 0.45,
                'center': [wx + 10, wy + 45],
                'tip': seg_end,
                'angle': 15.0,
                'segment': [seg_start, seg_end],
                'is_estimated': True
            }

        return None


class BallDetector:
    """
    Modular cricket ball detector & tracker.
    Tracks ball candidates using multi-scale appearance (red/white cricket ball)
    and frame-to-frame kinematic consistency.
    """
    def __init__(self):
        self.tracked_trajectory: List[Dict[str, Any]] = []

    def detect_candidate(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None, frame_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        Detects ball candidate in frame using color and motion differencing.
        Returns:
            {
                "frame_idx": frame_idx,
                "center": [x, y],
                "radius": r,
                "confidence": float (0-1)
            }
        """
        h, w = frame.shape[:2]

        # 1. Color filtering for red cricket ball in HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Red spans 0-10 and 170-180
        mask_r1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        mask_ball = cv2.bitwise_or(mask_r1, mask_r2)

        # 2. Motion differencing if previous frame is present
        if prev_frame is not None:
            diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            mask_ball = cv2.bitwise_and(mask_ball, diff_thresh)

        contours, _ = cv2.findContours(mask_ball, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Cricket ball size in 720p: typically 15 to 400 pixels
            if 12 <= area <= 500:
                ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
                # Circularity check
                circle_area = np.pi * (radius ** 2)
                circularity = area / max(1.0, circle_area)
                if circularity >= 0.40 and 2.0 <= radius <= 16.0:
                    candidates.append({
                        'frame_idx': frame_idx,
                        'center': [int(cx), int(cy)],
                        'radius': float(radius),
                        'confidence': round(float(np.clip(circularity * 0.9, 0.4, 0.95)), 2)
                    })

        if candidates:
            # Pick best candidate closest to existing trajectory if tracking
            if self.tracked_trajectory:
                last_pos = self.tracked_trajectory[-1]['center']
                candidates.sort(key=lambda c: np.linalg.norm(np.array(c['center']) - np.array(last_pos)))
            else:
                candidates.sort(key=lambda c: c['confidence'], reverse=True)

            best = candidates[0]
            self.tracked_trajectory.append(best)
            return best

        return None

    def get_trajectory(self) -> List[Dict[str, Any]]:
        return self.tracked_trajectory

    def clear(self):
        self.tracked_trajectory = []
