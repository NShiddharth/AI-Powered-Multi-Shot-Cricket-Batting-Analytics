import sys
sys.path.insert(0, '.')
import glob
from src.preprocessing.video_preprocessing import VideoPreprocessor
from src.pose_estimation.posture_tracker import VisionPostureTracker
import numpy as np

vids = sorted(glob.glob('data/reference_videos/cover_drive/*.avi'))
for vp in vids:
    p = VideoPreprocessor(vp)
    tracker = VisionPostureTracker()
    tracker.profile_video(vp)
    fms = []
    for idx, f in p.read_frames():
        lm = tracker.process_frame(f, idx, p.total_frames)
        if lm: fms.append(lm)
    p.release(); tracker.close()
    if fms:
        mid = fms[len(fms)//2]
        ls, rs = mid['left_shoulder'], mid['right_shoulder']
        dx, dy = rs[0]-ls[0], rs[1]-ls[1]
        ang = np.arctan2(dy, dx)*180/np.pi
        print(f"{vp.split('/')[-1]}: dx={dx:+.3f}, dy={dy:+.3f}, ang={ang:+.1f}, abs={abs(ang):.1f}")
