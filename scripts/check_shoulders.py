import sys
sys.path.insert(0, '.')
import json
from src.preprocessing.video_preprocessing import VideoPreprocessor
from src.pose_estimation.posture_tracker import VisionPostureTracker

import numpy as np

def inspect_video(path):
    p = VideoPreprocessor(path)
    tracker = VisionPostureTracker()
    tracker.profile_video(path)
    frames_lm = []
    for idx, frame in p.read_frames():
        lm = tracker.process_frame(frame, idx, p.total_frames)
        frames_lm.append(lm)
    p.release()
    tracker.close()
    
    print(f"\n--- {path} (total {len(frames_lm)} frames) ---")
    for f_idx in [len(frames_lm)//4, len(frames_lm)//2, 3*len(frames_lm)//4]:
        lm = frames_lm[f_idx]
        if lm:
            ls = lm['left_shoulder']
            rs = lm['right_shoulder']
            dx = rs[0] - ls[0]
            dy = rs[1] - ls[1]
            raw_angle = np.arctan2(dy, dx) * 180.0 / np.pi
            print(f"Frame {f_idx}: ls=({ls[0]:.2f}, {ls[1]:.2f}), rs=({rs[0]:.2f}, {rs[1]:.2f}), dx={dx:.2f}, dy={dy:.2f}, raw_angle={raw_angle:.1f}, abs={abs(raw_angle):.1f}")

inspect_video("data/reference_videos/straight_drive/straight_0101.avi")
inspect_video("data/reference_videos/straight_drive/straight_0102.avi")
inspect_video("data/reference_videos/straight_drive/straight_0104.avi")
