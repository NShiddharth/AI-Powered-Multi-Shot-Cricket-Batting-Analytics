import sys
sys.path.insert(0, '.')
import glob
from src.preprocessing.video_preprocessing import VideoPreprocessor
from src.pose_estimation.posture_tracker import VisionPostureTracker
import numpy as np

for shot in ['cover_drive', 'straight_drive', 'pull_shot']:
    vids = sorted(glob.glob(f'data/reference_videos/{shot}/*.avi'))
    print(f"\n{'='*20} {shot.upper()} {'='*20}")
    for vp in vids:
        p = VideoPreprocessor(vp)
        tracker = VisionPostureTracker()
        tracker.profile_video(vp)
        frames_lm = []
        for idx, frame in p.read_frames():
            lm = tracker.process_frame(frame, idx, p.total_frames)
            if lm:
                frames_lm.append(lm)
        p.release()
        tracker.close()
        
        if frames_lm:
            mid_lm = frames_lm[len(frames_lm)//2]
            ls = mid_lm['left_shoulder']
            rs = mid_lm['right_shoulder']
            nose = mid_lm['nose']
            # Check batter facing direction: is nose to the left or right of shoulders?
            sh_mid_x = (ls[0] + rs[0]) / 2.0
            facing = "facing_left" if nose[0] < sh_mid_x else "facing_right"
            print(f"{vp.split('/')[-1]}: frames={len(frames_lm)}, nose_x={nose[0]:.2f}, sh_mid_x={sh_mid_x:.2f}, {facing}")
